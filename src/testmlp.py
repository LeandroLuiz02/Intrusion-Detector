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

# Carregar e processar os dados de validação
dos_validation_file_path = '../attacks/validation/0-dos-candump-2024-07-10_184308.log'
dos_validation_data = load_and_process_data(dos_validation_file_path)

falsifying_validation_file_path = '../attacks/validation/1-falsifying-candump-2024-07-10_184439.log'
falsifying_validation_data = load_and_process_data(falsifying_validation_file_path)

fuzzing_validation_file_path = '../attacks/validation/2-fuzzing-candump-2024-07-10_184609.log'
fuzzing_validation_data = load_and_process_data(fuzzing_validation_file_path)

impersonation_validation_file_path = '../attacks/validation/3-impersonation-candump-2024-07-10_184739.log'
impersonation_validation_data = load_and_process_data(impersonation_validation_file_path)

validation_data = pd.concat([fuzzing_validation_data, dos_validation_data, falsifying_validation_data, impersonation_validation_data])

# Extrair características das novas mensagens
new_features = np.array(validation_data['message'].apply(extract_features).tolist())
print(new_features)

# Normalizar os novos dados
new_features = scaler.transform(new_features)

# Fazer previsões
y_pred_probs = model.predict(new_features)

# Ajuste do limiar de decisão
threshold = 0.7
y_pred = (y_pred_probs > threshold).astype("int32")

# Adicionar as previsões ao DataFrame original
validation_data['prediction'] = y_pred


# Converter rótulos reais para 0 e 1
y_true = np.array([1 if label == 'T' else 0 for label in validation_data['label']])
    
# Calcular a acurácia
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
    
# Calcular matriz de confusão e relatório de classificação
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Salvar as previsões em um novo arquivo
validation_data.to_csv('previsoes.csv', index=False, header=False)

# Exibir algumas das previsões
print(validation_data.head())