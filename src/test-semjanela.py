from utils import Filter, CommunicationMatrix, CANMsgFromline
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import re
from sys import argv
import pandas as pd

# Função para extrair características das mensagens CAN
def extract_features(message):
    parts = re.split('[#]', message)
    id_hex = parts[0]
    data_hex = parts[1] if len(parts) > 1 else ''

    id_int = int(id_hex, 16)
    data_bytes = [int(data_hex[i:i+2], 16) for i in range(0, len(data_hex), 2)]

    data_bytes += [0] * (8 - len(data_bytes))

    return [id_int] + data_bytes

def process_single_message(message):
    match = re.match(r'.* can0 (\S+) ([RT])', message)
    if match:
        message_part = match.group(1)
        label_part = match.group(2)
        data = pd.DataFrame({'message': [message_part], 'label': [label_part]})
        return data
    else:
        return pd.DataFrame({'message': [], 'label': []})

def process_single_message_nolabel(message):
    match = re.match(r'\(\d+\.\d+\)\s+can0\s+(\S+)', message)
    if match:
        message_part = match.group(1)
        data = pd.DataFrame({'message': [message_part]})
        return data
    else:
        return pd.DataFrame({'message': []})

# Carregar o modelo treinado
model = load_model('idsmodel.h5')
scaler = joblib.load('scaler.pkl')

# Ler o arquivo de ataques
attack_file = "../attacks/validation/3-impersonation-candump-2024-07-31_190254.log" if len(argv) == 1 else argv[1]
print(f"Lendo arquivo de ataques: {attack_file}")
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
    if msg is None: 
        continue
    if msg.label is None:
        print('Unknown label')
        continue

    # Testar a mensagem com o filtro
    filter_result = filter.test(msg)
    
    # Se o filtro classificar como normal, passar para o modelo MLP
    if filter_result == 'Normal':
        processedmsg = process_single_message_nolabel(line)
        extract = extract_features(processedmsg['message'][0])
        features = np.array(extract).reshape(1, -1)
        features_scaled = scaler.transform(features)
        y_pred_prob = model.predict(features_scaled)
        
        thresholdmlp = 0.7
        y_pred = (y_pred_prob > thresholdmlp).astype("int32")

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

# Calcular e imprimir métricas
total_msgs = true_positive + true_negative + false_positive + false_negative
print(f"Total de mensagens: {total_msgs}")
print(f"Total de mensagens certas: {true_positive + true_negative}")
print(f"Total de mensagens erradas: {false_positive + false_negative}")
print(f"Acurácia: {(true_positive + true_negative)/total_msgs}")
print(f"True Positive: {true_positive/total_msgs}")
print(f"True Negative: {true_negative/total_msgs}")
print(f"False Positive: {false_positive/total_msgs}")
print(f"False Negative: {false_negative/total_msgs}")