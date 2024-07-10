import re
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class CANDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def parse_log_file(file_path, max_len=8):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    labels = []

    for line in lines:
        if " T" in line:
            label = 1  # Ataque
        elif " R" in line:
            label = 0  # Normal
        else:
            continue

        # Extraindo a mensagem CAN (números hexadecimais após o #)
        match = re.search(r'#([A-Fa-f0-9]+)', line)
        if match:
            message = match.group(1)
            # Convertendo a mensagem CAN para uma lista de bytes
            byte_message = [int(message[i:i+2], 16) for i in range(0, len(message), 2)]
            # Padding para garantir que todas as mensagens tenham o mesmo comprimento
            while len(byte_message) < max_len:
                byte_message.append(0)
            byte_message = byte_message[:max_len]  # Truncar se for maior que max_len
            data.append(byte_message)
            labels.append(label)

    return data, labels

def load_datasets():
    normal_data, normal_labels = parse_log_file('../attacks/NORMAL_SAMPLES.txt')
    dos_data, dos_labels = parse_log_file('../attacks/DOS_ATCK.txt')
    falsifying_data, falsifying_labels = parse_log_file('../attacks/FALSIFYING_ATCK.txt')
    fuzzing_data, fuzzing_labels = parse_log_file('../attacks/FUZZING_ATCK.txt')
    impersonation_data, impersonation_labels = parse_log_file('../attacks/IMPERSONATION_ATCK.txt')

    data = normal_data + dos_data + falsifying_data + fuzzing_data + impersonation_data
    labels = normal_labels + dos_labels + falsifying_labels + fuzzing_labels + impersonation_labels

    return train_test_split(data, labels, test_size=0.2, random_state=42)

train_data, test_data, train_labels, test_labels = load_datasets()

train_dataset = CANDataset(train_data, train_labels)
test_dataset = CANDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, sequence_length):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.sequence_length = sequence_length

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.ReLU(),
            nn.Linear(512, self.sequence_length),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        sequence = self.model(gen_input)
        return sequence

class Discriminator(nn.Module):
    def __init__(self, num_classes, sequence_length):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.sequence_length = sequence_length

        input_dim = num_classes + sequence_length

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.validity_layer = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()  # Certificando que a saída é entre 0 e 1
        )
        self.aux_layer = nn.Linear(512, num_classes)

    def forward(self, sequence, labels):
        label_embedding = self.label_embedding(labels)
        d_in = torch.cat((sequence, label_embedding), -1)
        x = self.model(d_in)
        validity = self.validity_layer(x)
        label = self.aux_layer(x)
        return validity, label

# Configurações
latent_dim = 100
num_classes = 5
sequence_length = 8  # Exemplo de comprimento da sequência CAN

# Inicialização dos modelos
generator = Generator(latent_dim, num_classes, sequence_length)
discriminator = Discriminator(num_classes, sequence_length)
import torch.optim as optim

# Funções de perda
adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

# Otimizadores
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
import numpy as np

n_epochs = 200
sample_interval = 400
cuda = True if torch.cuda.is_available() else False

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

for epoch in range(n_epochs):
    for i, (sequences, labels) in enumerate(train_loader):

        batch_size = sequences.size(0)

        valid = torch.ones(batch_size, 1, requires_grad=False).cuda()
        fake = torch.zeros(batch_size, 1, requires_grad=False).cuda()

        real_sequences = sequences.cuda()
        labels = labels.cuda()

        optimizer_G.zero_grad()

        z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))).cuda()
        gen_labels = torch.LongTensor(np.random.randint(0, num_classes, batch_size)).cuda()

        gen_sequences = generator(z, gen_labels)

        validity, pred_label = discriminator(gen_sequences, gen_labels)

        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_pred, real_aux = discriminator(real_sequences, labels)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        fake_pred, fake_aux = discriminator(gen_sequences.detach(), gen_labels)
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch+1}/{n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            
correct = 0
total = 0

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.cuda()
        labels = labels.cuda()
        outputs, aux_outputs = discriminator(sequences, labels)
        _, predicted = torch.max(aux_outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Acurácia no conjunto de validação: {accuracy:.2f}%')