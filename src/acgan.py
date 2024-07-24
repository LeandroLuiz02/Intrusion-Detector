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

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from time import sleep

def testGan(dataloader, discriminator, opt = get_cmd_args(), cuda=True, print_results = True):

    # Teste
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i, (imgs, labels) in enumerate(dataloader):

        # Cria um tensor de ponto flutuante
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # Cria um tensor de inteiros longos
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        batch_size = imgs.shape[0]

        # Configuração das variáveis
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configuração dos inputs
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # Passa as imagens reais pelo discriminador para obter a validade
        real_pred, real_aux = discriminator(real_imgs)

        # Calcula a acuracia do descriminador
        pred = real_aux.data.cpu().numpy()
        gt = labels.data.cpu().numpy()
        # print(real_pred)
        # print(np.argmax(pred, axis=1))
        # print(gt)
        # print(np.argmax(pred, axis=1) == gt)
        # exit(0)

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

    sum = true_positive + true_negative + false_positive + false_negative
    if print_results:
        print(f"Accuracy: {(true_positive + true_negative)/sum}%, {true_positive + true_negative} of {sum}")
        print(f"True Positive: {true_positive/sum}%, {true_positive}")
        print(f"True Negative: {true_negative/sum}%, {true_negative}")
        print(f"False Positive: {false_positive/sum}%, {false_positive}")
        print(f"False Negative: {false_negative/sum}%, {false_negative}")
    return (true_negative + true_positive)/sum

def trainGan(datasets_and_types_of_attacks, opt = get_cmd_args(), model_file_name: str = 'model.pth'):
    # Usa o cuda se tiver disponivel
    cuda = True if torch.cuda.is_available() else False

    # Criação do Dataset e DataLoader
    dataset = CANDataset(datasets_and_types_of_attacks, opt, mirror_imgs=opt.mirror_img, transform=get_transform(opt.img_size), window_size=8, stride=4)

    len_dataset = len(dataset)
    train_size = int(0.70 * len_dataset)
    test_size = int(0.15 * len_dataset)
    val_size = len_dataset - train_size - test_size

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)

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

    current_acc = 0
    current_patience = 0

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(train_dataloader):

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

            # Passa as imagens geradas pelo gerador
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Passa as imagens reais pelo discriminador para obter a validade
            real_pred, real_aux = discriminator(real_imgs)

            # Calcula a perda total do discriminador como a média das perdas para imagens reais e falsas
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calcula a acuracia do descriminador
            pred = (np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                    if False else real_aux.data.cpu().numpy())
            gt = (np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                    if False else labels.data.cpu().numpy())

            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            # d_real_loss.backward()
            optimizer_D.step()

            # Resultados
            if i % 5 == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), 100 * d_acc, g_loss.item()))

        new_acc = testGan(val_dataloader, discriminator, opt, print_results=False)
        if new_acc - current_acc <= 0 or new_acc - current_acc < opt.tolerance: 
            current_patience += 1
        else:
            current_patience = 0

        if new_acc > current_acc:
            print('Saving model with validation accuracy: ', new_acc)
            current_acc = new_acc
            torch.save(discriminator.state_dict(), model_file_name)
            sleep(1)

        if current_patience > opt.patience:
            break

    print("Test accuracy:\n")
    testGan(test_dataloader, discriminator, opt)
