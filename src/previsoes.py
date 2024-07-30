import pandas as pd

# Carregar o arquivo de previsões
predictions_file_path = 'previsoes.csv'
predictions_data = pd.read_csv(predictions_file_path, header=None, names=['message', 'label', 'prediction'])

# Mapear rótulos e previsões para 0 e 1
label_mapping = {'R': 0, 'T': 1}
predictions_data['label'] = predictions_data['label'].map(label_mapping)

# Comparar rótulos reais com previsões
predictions_data['correct'] = predictions_data['label'] == predictions_data['prediction']

# Calcular acertos e erros
acertos = predictions_data['correct'].sum()
erros = len(predictions_data) - acertos

# Exibir resultados
print(f'Acertos: {acertos}')
print(f'Erros: {erros}')

# Exibir detalhes dos acertos e erros
print("\nDetalhes dos Acertos:")
print(predictions_data[predictions_data['correct']])

print("\nDetalhes dos Erros:")
print(predictions_data[~predictions_data['correct']])
