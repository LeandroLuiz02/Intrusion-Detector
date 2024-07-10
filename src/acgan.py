import torch
import torch.nn as nn

# Defina as classes Generator e Discriminator aqui (ou importe se estiverem em um módulo separado)
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
            nn.Sigmoid()
        )
        self.aux_layer = nn.Linear(512, num_classes)

    def forward(self, sequence, labels):
        label_embedding = self.label_embedding(labels)  # Obtém a incorporação de rótulo
        d_in = torch.cat((sequence, label_embedding), -1)  # Concatena ao longo da última dimensão
        x = self.model(d_in)  
        validity = self.validity_layer(x)
        label = self.aux_layer(x)
        return validity, label

# Configurações
latent_dim = 100
num_classes = 5  # Número de classes que você tem (ajuste conforme necessário)
sequence_length = 8  # Comprimento da sequência CAN

# Inicialização dos modelos
generator = Generator(latent_dim, num_classes, sequence_length)
discriminator = Discriminator(num_classes, sequence_length)

# Carregue o modelo do ACGAN, mapeando para CPU se necessário
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(latent_dim, num_classes, sequence_length).to(device)
discriminator = Discriminator(num_classes, sequence_length).to(device)

# Ajuste o caminho do seu modelo conforme necessário
generator.load_state_dict(torch.load('./acgan/generator.pth', map_location=device))
discriminator.load_state_dict(torch.load('./acgan/discriminator.pth', map_location=device))

# Modo de avaliação
generator.eval()
discriminator.eval()