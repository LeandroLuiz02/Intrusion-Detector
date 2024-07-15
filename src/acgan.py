# Imports
import os
import numpy as np
import math
import sys
from PIL import Image, ImageDraw
from canDataset import *
from ganClasses import Generator, Discriminator
from parser import opt
from utils import weights_init_normal, get_transform

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

opt = get_cmd_args()

# Usa o cuda se tiver disponivel
cuda = True if torch.cuda.is_available() else False

# Criação do Dataset e DataLoader
dataset = CANDataset([
    ('../attacks/DOS_ATCK.txt', DOS_MSG),
    ('../attacks/FUZZING_ATCK.txt', FUZZY_MSG),
    ('../attacks/FALSIFYING_ATCK.txt', FALS_MSG),
    ('../attacks/IMPERSONATION_ATCK.txt', IMP_MSG),
], opt, mirror_imgs=True, transform=get_transform(opt.img_size))
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

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
if opt.test:
    generator.load_state_dict(torch.load('./generator.pth'))
    discriminator.load_state_dict(torch.load('./discriminator.pth'))
else:
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
prev_acc = 0 # Acuracia anterior
acc = 0 # Acuracia
cnt = 0 # Contador, para salvar o modelo com a melhor acuracia

# Teste
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

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

        if not opt.test:
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

        if not opt.test:
            optimizer_D.zero_grad()

            # Passa as imagens geradas pelo gerador
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Passa as imagens reais pelo discriminador para obter a validade
        real_pred, real_aux = discriminator(real_imgs)

        # Calcula a perda total do discriminador como a média das perdas para imagens reais e falsas
        if not opt.test:
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
            d_loss = (d_real_loss + d_fake_loss) / 2

        # Calcula a acuracia do descriminador
        pred = (np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0) 
                if not opt.test else real_aux.data.cpu().numpy())
        gt = (np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                if not opt.test else labels.data.cpu().numpy())

        d_acc = np.mean(np.argmax(pred, axis=1) == gt)
        acc = (d_acc + acc*i) / (i+1)

        if not opt.test:
            d_loss.backward()
            optimizer_D.step()

        if opt.test:
            for pr, ex in zip(np.argmax(pred, axis=1), gt):
                if pr == ex:
                    if pr == 0:
                        true_negative += 1
                    else:
                        true_positive += 1
                else:
                    if pr == 0:
                        false_negative += 1
                    else:
                        false_positive += 1
        if i == len(dataloader) - 1:
            if opt.test:
                sum = true_positive + true_negative + false_positive + false_negative
                print(f"True Positive: {true_positive/sum}")
                print(f"True Negative: {true_negative/sum}")
                print(f"False Positive: {false_positive/sum}")
                print(f"False Negative: {false_negative/sum}")
                exit(0)

            # Fim da iteração
            if prev_acc < acc:
                cnt = 0
                prev_acc = acc
                print('Saving model with accuracy: ', acc)
                torch.save(generator.state_dict(), 'generator.pth')
                torch.save(discriminator.state_dict(), 'discriminator.pth')
            else:
                # Se a acuracia não melhorar, incrementa o contador
                # Se o contador chegar ao valor de patience, para o treinamento
                print(f'Accuracy did not improve: {acc} <= {prev_acc}')
                cnt += 1
                if cnt == opt.patience:
                    print('Early stopping')
                    exit(0)

        # Resultados
        if i % 5 == 0:
            print("[Epoch %d/%d] [Batch %d/%d]" % (epoch, opt.n_epochs, i, len(dataloader)), end='\n' if opt.test else '')
            if not opt.test: print("[D loss: %f, acc: %d%%] [G loss: %f]" % (d_loss.item(), 100 * d_acc, g_loss.item()))

# torch.save(generator.state_dict(), 'generator.pth')
# torch.save(discriminator.state_dict(), 'discriminator.pth')
