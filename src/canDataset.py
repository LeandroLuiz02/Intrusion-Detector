from image import *
from typing import List, Tuple
from utils import *
from image import *
import torch
from torchvision import transforms
import numpy as np

class DatasetClass():
    def __init__(self):
        self.imgs = []
        self.labels = []
    def __len__(self): return len(self.imgs)
    def insert(self, img, label):
        self.imgs.append(img)
        self.labels.append(label)

    def oversample(self, max_size):
        ratio = max_size // len(self.imgs)
        rest = max_size % len(self.imgs)
        self.imgs = self.imgs * ratio + self.imgs[:rest]
        self.labels = self.labels * ratio + self.labels[:rest]

# Dataset personalizado
class CANDataset():

    def __init__(self, datasets_and_type_of_attacks : List[Tuple[str,str]], opt, transform=None):
        '''
        Args:
        - datasets_and_type_of_attacks (List[str]): Lista de tuplas de strings, uma path e o tipo de ataque dele.
        - window_size (int): Tamanho da janela de mensagens.
        - stride (int): Tamanho do passo da janela.
        - mirror_imgs (bool): Se deve espelhar as imagens.
        - transform (callable, optional): Transformação a ser aplicada nas imagens.
        '''
        self.msgs = []
        # Contains the images and labels of each class of the dataset
        self.samples = {
            NORMAL_MSG: DatasetClass(),
            DOS_MSG: DatasetClass(),
            FUZZY_MSG: DatasetClass(),
            FALS_MSG: DatasetClass(),
            IMP_MSG: DatasetClass(),
        }
        self.opt = opt
        self.window_size = opt.window_size
        self.stride = opt.stride
        self.mirror_imgs = opt.mirror_img
        self.transform = transform
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
                for i in range(0, len(self.msgs), self.stride):
                    current_msgs = self.msgs[i:i+self.window_size]
                    has_attack = any(msg.label != MSGS_TYPES[NORMAL_MSG] for msg in current_msgs)

                    # Número máximo de ataques por classe
                    if has_attack and len(self.samples[type_of_attack]) >= opt.dataset_max_size:
                        break
                    # Número máximo de mensagens normais
                    elif not has_attack and len(self.samples[NORMAL_MSG]) >= opt.dataset_max_size:
                        continue

                    img = create_can_image(current_msgs, mirror=self.mirror_imgs)
                    # Aplica a transformação na imagem
                    if self.transform is not None:
                        img = self.transform(img)

                    # Adiciona a imagem e a label ao dataset da sua classe específica
                    msg_label = type_of_attack if has_attack else NORMAL_MSG
                    self.samples[msg_label].insert(img, MSGS_TYPES[msg_label])

                self.msgs = []
        if opt.pregenerate_imgs:
            if not self.opt.test: self.oversample(500)
            self.imgs = []
            self.labels = []
            for k, class_data in self.samples.items():
                print(k)
                print(len(class_data))
                self.imgs += class_data.imgs
                self.labels += class_data.labels

    def __len__(self): return len(self.msgs) // self.stride if not self.opt.pregenerate_imgs else len(self.imgs)

    def __getitem__(self, idx):
        if self.opt.pregenerate_imgs:
            return self.get_item_with_preprocessing(idx)
        return self.get_item_without_preprocessing(idx)

    def oversample(self, max_size):
        if self.opt.pregenerate_imgs:
            for class_data in self.samples.values():
                if len(class_data) > max_size or len(class_data) == 0: continue
                class_data.oversample(max_size)
        else:
            print('Oversampling not supported for non pregenerated images')
            exit(0)

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
