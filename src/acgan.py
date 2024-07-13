# Imports
import os
import numpy as np
import math
import sys
from PIL import Image, ImageDraw
from canDataset import *
from ganClasses import Generator, Discriminator
from parser import opt
from utils import weights_init_normal

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# Usa o cuda se tiver disponivel
cuda = True if torch.cuda.is_available() else False

# Transformações
transform = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,], [0.5,])
])

# Criação do Dataset e DataLoader
dataset = CANDataset([
    ('../attacks/NORMAL_SAMPLES.txt', NORMAL_MSG),
    ('../attacks/DOS_ATCK.txt', DOS_MSG),
    ('../attacks/FUZZING_ATCK.txt', FUZZY_MSG),
    ('../attacks/FALSIFYING_ATCK.txt', FALS_MSG),
    ('../attacks/IMPERSONATION_ATCK.txt', IMP_MSG),
], mirror_imgs=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Funções de perda

# Binary Cross Entropy Loss, função de perda binária.
adversarial_loss = torch.nn.BCELoss()

# Cross Entropy Loss, função de perda de entropia cruzada. Usada auxiliarmente.
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Inicializações

# Inializando o gerador e o descriminador
generator = Generator(opt)
discriminator = Discriminator(opt)

# Verifica se o CUDA está disponível
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Inicializa os pesos
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Otimizadores

# Otimizador para o gerador
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# Otimizador para o discriminador
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Cria um tensor de ponto flutuante
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# Cria um tensor de inteiros longos
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# Treinamento

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Configuração das variáveis
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configuração dos inputs
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # Treino do Gerador

        optimizer_G.zero_grad()

        # Amostra ruído e rótulos aleatórios para alimentar o gerador
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Gera um lote de imagens a partir do ruído e dos rótulos
        gen_imgs = generator(z, gen_labels)

        # Passa as imagens geradas pelo discriminador para obter a validade e os rótulos previstos
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        #  Calcula os gradientes da perda em relação aos parâmetros do gerador e atualiza
        g_loss.backward()
        optimizer_G.step()

        # Treino do Descriminador

        optimizer_D.zero_grad()

        # Passa as imagens reais pelo discriminador para obter a validade
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Passa as imagens geradas pelo discriminador
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Calcula a perda total do discriminador como a média das perdas para imagens reais e falsas
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calcula a acuracia do descriminador
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        # Resultados
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
