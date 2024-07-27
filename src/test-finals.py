from utils import Filter, CommunicationMatrix, CANMsgFromline
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import re
from sys import argv

#NOT WORKING, THE MODEL DOES NOT SAVE PROPRIETLY
#PROBABLY THE MODEL IS NOT SAVING THE WEIGHTS

# Função para extrair características das mensagens CAN
def extract_features(message):
    parts = re.split('[#]', message)
    id_hex = parts[0]
    data_hex = parts[1] if len(parts) > 1 else ''

    id_int = int(id_hex, 16)
    data_bytes = [int(data_hex[i:i+2], 16) for i in range(0, len(data_hex), 2)]

    data_bytes += [0] * (8 - len(data_bytes))

    return [id_int] + data_bytes

# Carregar o modelo treinado
model = load_model('idsmodel.h5')

# Carregar o escalador e usar os dados de treinamento para ajustá-lo
scaler = joblib.load('scaler.pkl')
X_train_load = np.load('X_train.npy')
X_train = scaler.fit_transform(X_train_load)

# Ler o arquivo de ataques
attack_file = "../attacks/validation/0-dos-candump-2024-07-10_184308.log" if len(argv) == 1 else argv[1]
print(f"Lendo arquivo de ataques: {attack_file}")
#print(attack_file)
with open(attack_file, 'r') as file:
    text = file.read()

# Criar e configurar o filtro
filter = Filter(CommunicationMatrix('./communication_matrix.json'), threshold=3, tolerance=0.03, enable_time=False)

# Inicializar contadores para métricas
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# Processar mensagens do arquivo
msgs = text.split('\n')
for line in msgs:
    msg = CANMsgFromline(line)
    if msg is None: break
    if msg.label is None:
        print('Unknown label')
        continue
    #if msg is None:
    #    print(f"Mensagem inválida: {line}")
    #    continue
    #if msg.label is None:
    #    print(f"Rótulo desconhecido para a mensagem: {line}")
    #    continue
    #print(f"Processando mensagem: {msg.message} com rótulo: {msg.label}")

    # Testar a mensagem com o filtro
    filter_result = filter.test(msg)
    
    # Se o filtro classificar como normal, passar para o modelo MLP
    if filter_result == 'Normal':
        features = np.array(extract_features(msg.message)).reshape(1, -1)
        features_scaled = scaler.transform(features)
        y_pred_prob = model.predict(features_scaled)
        
        threshold = 0.7
        y_pred = (y_pred_prob > threshold).astype("int32")
        
        if y_pred == 1:
            ids_result = 'Attack'
        else:
            ids_result = 'Normal'
    else:
        ids_result = 'Attack'

    # Atualizar contadores de métricas
    if ids_result == 'Attack' and msg.label == 'Attack':
        true_positive += 1
    elif ids_result == 'Normal' and msg.label == 'Normal':
        true_negative += 1
    elif ids_result == 'Attack' and msg.label == 'Normal':
        false_positive += 1
    elif ids_result == 'Normal' and msg.label == 'Attack':
        false_negative += 1
    else:
        print('Unknown label or result')
        exit(1)

# Calcular e imprimir métricas
total_msgs = len(msgs) - msgs.count('')
print(f"True Positive: {true_positive/total_msgs}")
print(f"True Negative: {true_negative/total_msgs}")
print(f"False Positive: {false_positive/total_msgs}")
print(f"False Negative: {false_negative/total_msgs}")
