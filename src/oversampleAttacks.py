import pandas as pd
from sklearn.utils import resample
import os

def load_data(file_paths):
    dataframes = []
    for path in file_paths:
        df = pd.read_csv(path, header=None, names=['message'])
        dataframes.append(df)
    return dataframes

def find_min_attack_count(dataframes):
    min_count = float('inf')
    for df in dataframes:
        count = df['message'].str.endswith(' T').sum()
        if count < min_count:
            min_count = count
    return min_count

def balance_dataframes(dataframes, min_count, window_size = 1000):
    balanced_dfs = []
    for df in dataframes:
        attack_df = df[df['message'].str.endswith(' T')]
        if len(attack_df) > min_count:
            num_windows = min_count // window_size
            attack_df = pd.concat([attack_df] * num_windows, ignore_index=True)
            remainder = min_count % window_size
            if remainder > 0:
                attack_df = pd.concat([attack_df, attack_df.sample(remainder, random_state=42)], ignore_index=True)
        balanced_dfs.append(attack_df)
    return balanced_dfs


def save_balanced_dataframes(balanced_dfs, file_paths):
    for i, df in enumerate(balanced_dfs):
        original_file_path = file_paths[i]
        file_dir, file_name = os.path.split(original_file_path)
        file_name_wo_ext, file_ext = os.path.splitext(file_name)
        balanced_file_path = os.path.join(file_dir, f"{file_name_wo_ext}_balanced{file_ext}")
        df.to_csv(balanced_file_path, index=False, header=False)
        print(f"Balanced file saved as: {balanced_file_path}")

# Caminhos dos arquivos 
file_paths = ["../attacks/DOS_ATCK.txt",
              "../attacks/FALSIFYING_ATCK.txt",
              "../attacks/FUZZING_ATCK.txt",
              "../attacks/IMPERSONATION_ATCK.txt"]

# Carregar os dados
dataframes = load_data(file_paths)

# Encontrar o menor número de ataques entre os tipos
min_count = find_min_attack_count(dataframes)

# Balancear os DataFrames
balanced_dfs = balance_dataframes(dataframes, min_count)

# Salvar os DataFrames balanceados
save_balanced_dataframes(balanced_dfs, file_paths)