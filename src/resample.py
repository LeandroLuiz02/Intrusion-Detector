import pandas as pd

from sklearn.utils import resample
import os

# Função para carregar os arquivos e contar as mensagens
def load_and_count_messages(file_paths):
    counts = []
    dataframes = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        messages = [line.strip() for line in lines]
        df = pd.DataFrame(messages, columns=['message'])

        count_normal = df['message'].str.endswith(' R').sum()
        count_attack = df['message'].str.endswith(' T').sum()

        counts.append((count_normal, count_attack))
        dataframes.append(df)

        print(f"File: {file_path} - Normal: {count_normal}, Attack: {count_attack}")

    return dataframes, counts

# Função para balancear os datasets com oversampling
def balance_datasets(dataframes, counts):
    balanced_dfs = []

    for df, (count_normal, count_attack) in zip(dataframes, counts):
        df_normal = df[df['message'].str.endswith(' R')]
        df_attack = df[df['message'].str.endswith(' T')]

        if count_normal > count_attack:
            df_attack_oversampled = resample(df_attack, replace=True, n_samples=count_normal, random_state=42)
            df_balanced = pd.concat([df_normal, df_attack_oversampled])
        else:
            df_normal_oversampled = resample(df_normal, replace=True, n_samples=count_attack, random_state=42)
            df_balanced = pd.concat([df_normal_oversampled, df_attack])

        # Embaralhar as mensagens
        # df_balanced = shuffle(df_balanced, random_state=42)
        # balanced_dfs.append(df_balanced)

        # Contar novamente as mensagens após balanceamento
        count_normal_balanced = df_balanced['message'].str.endswith(' R').sum()
        count_attack_balanced = df_balanced['message'].str.endswith(' T').sum()

        print(f"After balancing - Normal: {count_normal_balanced}, Attack: {count_attack_balanced}")

    return balanced_dfs

# Função para verificar se os datasets estão balanceados no total
def verify_balance(balanced_dfs):
    total_normal = 0
    total_attack = 0

    for df in balanced_dfs:
        messages = df['message']
        total_normal += messages.str.endswith(' R').sum()
        total_attack += messages.str.endswith(' T').sum()

    print(f"Total Normal: {total_normal}, Total Attack: {total_attack}")
    return total_normal, total_attack

# Caminhos dos arquivos CSV
file_paths = ["/content/drive/MyDrive/UFPE/2024.1/RA/DOS_ATCK.txt",
              "/content/drive/MyDrive/UFPE/2024.1/RA/FALSIFYING_ATCK.txt",
              "/content/drive/MyDrive/UFPE/2024.1/RA/FUZZING_ATCK.txt",
              "/content/drive/MyDrive/UFPE/2024.1/RA/IMPERSONATION_ATCK.txt"]

# Carregar os arquivos e contar as mensagens
dataframes, counts = load_and_count_messages(file_paths)

# Balancear os datasets
balanced_dfs = balance_datasets(dataframes, counts)

# Verificar o balanceamento total
total_normal, total_attack = verify_balance(balanced_dfs)

# Salvar os datasets balanceados com novos nomes
for i, df in enumerate(balanced_dfs):
    original_file_path = file_paths[i]
    file_dir, file_name = os.path.split(original_file_path)
    file_name_wo_ext, file_ext = os.path.splitext(file_name)
    balanced_file_path = os.path.join(file_dir, f"{file_name_wo_ext}_balanced{file_ext}")
    df.to_csv(balanced_file_path, index=False, header=False)
    print(f"Balanced file saved as: {balanced_file_path}")