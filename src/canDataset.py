from image import *
from typing import List, Tuple
from utils import *
from image import *
import torch
from torchvision import transforms
import numpy as np

# Dataset personalizado
class CANDataset():
    def __init__(self, datasets_and_type_of_attacks : List[Tuple[str,str]], opt, window_size=5, stride=3, mirror_imgs=False, transform=None):
        '''
        Args:
        - datasets_and_type_of_attacks (List[str]): Lista de tuplas de strings, uma path e o tipo de ataque dele.
        - window_size (int): Tamanho da janela de mensagens.
        - stride (int): Tamanho do passo da janela.
        - mirror_imgs (bool): Se deve espelhar as imagens.
        - transform (callable, optional): Transformação a ser aplicada nas imagens.
        '''
        self.msgs = []
        self.imgs = []
        self.labels = []
        self.opt = opt
        self.window_size = window_size
        self.stride = stride
        self.mirror_imgs = mirror_imgs
        self.transform = transform
        self.cnt = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 }
        for dataset, type_of_attack in datasets_and_type_of_attacks:
            print('Loading dataset:', dataset, 'with attack type:', type_of_attack)
            # Carrega o dataset
            with open(dataset, 'r') as file:
                text = file.read()
            for i, line in enumerate(text.split('\n'), 1):
                msg = CANMsgFromline(line)
                if msg is None: break
                msg.label = MSGS_TYPES[type_of_attack] if msg.label != 'Normal' else MSGS_TYPES[NORMAL_MSG]
                self.msgs.append(msg)
            if opt.pregenerate_imgs:
                for i in range(0, len(self.msgs), stride):
                    current_msgs = self.msgs[i:i+window_size]
                    has_attack = any(msg.label != MSGS_TYPES[NORMAL_MSG] for msg in current_msgs)

                    # Número máximo de ataques por classe
                    if has_attack:
                        if self.cnt[MSGS_TYPES[type_of_attack]] >= opt.dataset_max_size:
                            break
                        self.cnt[MSGS_TYPES[type_of_attack]] += 1
                    # Número máximo de mensagens normais
                    else:
                        if self.cnt[MSGS_TYPES[NORMAL_MSG]] >= opt.dataset_max_size:
                            continue
                        self.cnt[MSGS_TYPES[NORMAL_MSG]] += 1

                    img = create_can_image(current_msgs, mirror=mirror_imgs)
                    # Aplica a transformação na imagem
                    if self.transform is not None:
                        img = self.transform(img)
                    # Adiciona a imagem do dataset
                    self.imgs.append(img)
                    # Adiciona a label desta imagem ao dataset
                    self.labels.append(MSGS_TYPES[type_of_attack] if has_attack else MSGS_TYPES[NORMAL_MSG])

                self.msgs = []

    def __len__(self): return len(self.msgs) // self.stride if not self.opt.pregenerate_imgs else len(self.imgs)

    def __getitem__(self, idx):
        if self.opt.pregenerate_imgs:
            return self.get_item_with_preprocessing(idx)
        return self.get_item_without_preprocessing(idx)

    def get_item_without_preprocessing(self, idx):
        idx *= self.stride
        wind = self.msgs[idx : (idx+self.window_size)]
        img = create_can_image(wind, mirror=self.mirror_imgs)
        if self.transform is not None:
            img = self.transform(img)
        label = MSGS_TYPES[NORMAL_MSG]
        for msg in wind:
            if msg.label != MSGS_TYPES[NORMAL_MSG]:
                label = msg.label
                break
        return img, label

    def get_item_with_preprocessing(self, idx): return self.imgs[idx], self.labels[idx]

# # Exemplo de uso
#
# # Transformação a ser aplicada nas imagens
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5,], [0.5,])
# ])
#
# # Cria o dataset
# dataset = CANDataset([
#     ('../attacks/test.txt', NORMAL_MSG),
# ], mirror_imgs=True, transform=transform)
#
# # Exibe o tamanho do dataset
# print(len(dataset))
#
# # Exibe um exemplo do dataset
# img, label = dataset[0]
# print(img.shape, label)
#
# print(len(img))
