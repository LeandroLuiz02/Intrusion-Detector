import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold

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


# Carregar e processar os dados de Fuzzing Attack
fuzzing_file_path = '../attacks/FUZZING_ATCK.txt'
fuzzing_data = load_and_process_data(fuzzing_file_path)

# Carregar e processar os dados de DOS Attack
dos_file_path = '../attacks/DOS_ATCK.txt'
dos_data = load_and_process_data(dos_file_path)

# Carregar e processar os dados de Falsifying Attack
falsifying_file_path = '../attacks/FALSIFYING_ATCK.txt'
falsifying_data = load_and_process_data(falsifying_file_path)

# Carregar e processar os dados de Falsifying Attack
impersonation_file_path = '../attacks/IMPERSONATION_ATCK.txt'
impersonation_data = load_and_process_data(impersonation_file_path)

# Combinar os dados de Fuzzing e DOS Attack
training_data = pd.concat([fuzzing_data, dos_data, falsifying_data, impersonation_data])
training_features = np.array(training_data['message'].apply(extract_features).tolist())
training_labels = np.array([1 if label == 'T' else 0 for label in training_data['label']])

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(training_features, training_labels, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Aplicar SMOTE para balancear o conjunto de treino
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Original class distribution:", Counter(y_train))
print("Resampled class distribution:", Counter(y_train_resampled))

# Construção do modelo com Dropout e regularização L2
model = Sequential()
model.add(Dense(256, input_dim=X_train_resampled.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilação do modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Configurar o callback de EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Validação cruzada
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kfold.split(X_train_resampled, y_train_resampled):
    X_train_fold, X_val_fold = X_train_resampled[train_index], X_train_resampled[val_index]
    y_train_fold, y_val_fold = y_train_resampled[train_index], y_train_resampled[val_index]
    
    # Treinamento do modelo
    history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=100, batch_size=128, callbacks=[early_stopping])
    
# Avaliação do modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
y_pred_probs = model.predict(X_test)

# Ajuste do limiar de decisão
threshold = 0.7
y_pred = (y_pred_probs > threshold).astype("int32")

# Matriz de confusão e relatório de classificação
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Gráficos de precisão e perda durante o treinamento e validação
plt.figure(figsize=(12, 4))

# Gráfico de precisão
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

# Gráfico de perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.show()

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

validation_features = np.array(validation_data['message'].apply(extract_features).tolist())
validation_labels = np.array([1 if label == 'T' else 0 for label in validation_data['label']])

# Normalizar os dados de validação
validation_features = scaler.transform(validation_features)

# Avaliar o modelo nos dados de validação
validation_loss, validation_accuracy = model.evaluate(validation_features, validation_labels)
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')

# Fazer previsões nos dados de validação
validation_predictions_probs = model.predict(validation_features)
validation_threshold = 0.7
validation_predictions = (validation_predictions_probs > validation_threshold).astype(int).flatten()

# Imprimir relatório de classificação
print(classification_report(validation_labels, validation_predictions, target_names=['Normal', 'Attack']))

# Imprimir matriz de confusão
print(confusion_matrix(validation_labels, validation_predictions))

cm = confusion_matrix(validation_labels, validation_predictions)
cm_df = pd.DataFrame(cm, index=['Normal', 'Attack'], columns=['Predicted Normal', 'Predicted Attack'])
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()