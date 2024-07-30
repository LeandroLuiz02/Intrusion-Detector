import pandas as pd
import re
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Função para carregar e processar dados de um arquivo de texto
def load_and_process_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    messages = []
    labels = []
    for line in lines:
        match = re.match(r'.* can0 (\S+) ([RT])', line)
        if match:
            message = match.group(1)
            label = match.group(2)
            messages.append(message)
            labels.append(label)

    data = pd.DataFrame({'message': messages, 'label': labels})

    return data

# Função para extrair características das mensagens CAN
def extract_features(message):
    parts = re.split('[#]', message)
    id_hex = parts[0]
    data_hex = parts[1] if len(parts) > 1 else ''

    id_int = int(id_hex, 16)
    data_bytes = [int(data_hex[i:i+2], 16) for i in range(0, len(data_hex), 2)]

    data_bytes += [0] * (8 - len(data_bytes))

    return [id_int] + data_bytes

# Carregar o modelo e o scaler
model = load_model('idsmodel.h5')
scaler = joblib.load('scaler.pkl')

# Carregar e processar novos dados de mensagens
new_file_path = '../attacks/validation/3-impersonation-candump-2024-07-10_184739.log'
new_data = load_and_process_data(new_file_path)

# Extrair características das novas mensagens
new_features = np.array(new_data['message'].apply(extract_features).tolist())

# Normalizar os novos dados
new_features = scaler.transform(new_features)

# Fazer previsões
y_pred_probs = model.predict(new_features)

# Ajuste do limiar de decisão
threshold = 0.7
y_pred = (y_pred_probs > threshold).astype("int32")

# Adicionar as previsões ao DataFrame original
new_data['prediction'] = y_pred


# Converter rótulos reais para 0 e 1
y_true = np.array([1 if label == 'T' else 0 for label in new_data['label']])
    
# Calcular a acurácia
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
    
# Calcular matriz de confusão e relatório de classificação
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Salvar as previsões em um novo arquivo
new_data.to_csv('previsoes.csv', index=False, header=False)

# Exibir algumas das previsões
print(new_data.head())