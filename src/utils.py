from operator import is_
import torch
import torchvision.transforms as transforms
from ganClasses import Discriminator
from torch.autograd import Variable
from image import create_can_image
from utils import *
from typing import List
import numpy as np

NORMAL_MSG='Normal'
DOS_MSG='DoS'
FUZZY_MSG='Fuzzy'
FALS_MSG='Falsifying'
IMP_MSG='Impersonating'

# Classificação das mensagens
MSGS_TYPES = {
    NORMAL_MSG: 0,
    DOS_MSG: 1,
    FUZZY_MSG: 2,
    FALS_MSG: 3,
    IMP_MSG: 4,
}

# Valores para o tipo de mensagem
MSGS_TYPES_VALUES = {
    0: NORMAL_MSG,
    1: DOS_MSG,
    2: FUZZY_MSG,
    3: FALS_MSG,
    4: IMP_MSG,
}

# Transformações
get_transform = lambda size: transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,], [0.5,])
    ])

# Funções de perda

# Binary Cross Entropy Loss, função de perda binária.
adversarial_loss = torch.nn.BCELoss()

# Cross Entropy Loss, função de perda de entropia cruzada. Usada auxiliarmente.
auxiliary_loss = torch.nn.CrossEntropyLoss()

def get_cmd_args():
    from parser import opt
    return opt

# Função para ser utilizada com o método apply de PyTorch
def weights_init_normal(m):
    import torch
    classname = m.__class__.__name__
    # Verifica se a classe é convolucional
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    # Verifica se a classe é BatchNorm2d
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def setseed(seed):
    import random
    import torch
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class CommunicationMatrix():
    def __init__(self, file_path):
        import json
        with open(file_path, 'r') as f:
            self.matrix = json.load(f)

    def __str__(self):
        # TODO: Implement this
        return ""

class CANMessage():
    def __init__(self, time_stamp, id, payload, p_len, label = None):
        self.time_stamp = time_stamp
        self.id = id
        self.payload = payload
        self.p_len = p_len
        self.label = label

    def __str__(self):
        return f"\
    ===============\n\
    ID: {self.id}\n\
    Time Stamp: {self.time_stamp}\n\
    Payload Length: {self.p_len}\n\
    Payload: {self.payload}\n\
    Label: {self.label if self.label is not None else 'Unknown'}"

def CANMsgFromline(line : str):
    from math import ceil
    if line:
        words = line.split(' ')
        time_stamp = words[0][1:-1]
        id, _, payload = words[2].partition('#')
        label = ('Attack' if words[3] == 'T' else 'Normal') if len(words) == 4 else None
        return CANMessage(time_stamp, id, int(payload, 16), ceil(len(payload)/2), label)

class Filter():
    def __init__(self, comm_matrix : CommunicationMatrix,
                 window_size = 8, threshold = 3, enable_time = True, tolerance = 0.04):
        self.comm_matrix = comm_matrix
        # Just some random big negative number to prevent the
        # first message from being classified as an attack
        self.prev_msg_time = {}
        for id in self.comm_matrix.matrix:
            self.prev_msg_time[id] = -0xFFFF
        # Store if the last two messages were attacks or not
        self.prev_msg_label = 'Normal'
        self.acc = 0
        self.windown_size = window_size
        self.enable_time = enable_time
        self.threshold = threshold
        self.tolerance = tolerance

    def check_id_exists(self, msg : CANMessage):
        return msg.id in self.comm_matrix.matrix

    def check_payload_compatible(self, msg):
        def get_val_at_bit_interval(signal, start, end):
            mask = 0xFFFFFFFFFFFFFFFF # 64 bits, maximum length is 8 bytes
            # Bring the start bit to the beginning
            signal = signal >> start
            # Create a mask with the desired length and invert it's bits
            mask = (mask << (end - start)) ^ mask
            # Apply the mask to the signal
            return signal & mask

        def calculate_payload_vals(payload, id_info):
            s = 0 # Start index
            e = 1 # End index
            inter = 'bit_interval' # Just to make the code more readable
            return [
              # Get the value at the bit interval, apply the offset and factor
              (get_val_at_bit_interval(payload, signal[inter][s], signal[inter][e]+1) + signal['offset']) * signal['factor']
              for signal in id_info['signals']
            ]

        id_info = self.comm_matrix.matrix[msg.id]

        # Check if the payload length is compatible
        is_len_compatible = id_info['length'] >= msg.p_len
        if not is_len_compatible:
            return False

        # Check if the signal values are compatible, all signals must be in the range
        signal_values = calculate_payload_vals(msg.payload, id_info)
        return all([
            (True if (signal_values[i] >= signal['min'] and signal_values[i] <= signal['max']) else False)
            for i, signal in enumerate(id_info['signals']) 
        ])

    def check_is_in_time(self, msg):
        time = float(msg.time_stamp)
        is_in_time = True
        # Only check if the previous message was normal, possible attack coming
        if self.prev_msg_label == 'Normal': 
            is_in_time = (time - self.prev_msg_time[msg.id]) > self.tolerance
        return is_in_time

    def test(self, msgs : List[CANMessage], debug=False):
        # Check if the window size is correct
        if len(msgs) != self.windown_size:
            return None

        t = 0
        seen_ids = dict()
        for i, m in enumerate(msgs):
            if debug: print(f"Testing message {i+1} of the window")
            if self.enable_time:
                #Check if some other message with the same id of the current message appeared before
                # and if the time difference is less than the tolerance
                if m.id in seen_ids and float(m.time_stamp) - seen_ids[m.id] < self.tolerance:
                    if debug:
                        print(f"Time difference between messages with {m.id} is less than the tolerance")
                        print(f"Time difference: {float(m.time_stamp) - seen_ids[m.id]}")
                    t += 1
                else:
                    if debug: print(f"Adding id {m.id} to the seen ids")
                    seen_ids[m.id] = float(m.time_stamp)

            # Check for valid id
            if not self.check_id_exists(m):
                if debug: print("Invalid id")
                t += 1
            # Check for valid payload
            elif not self.check_payload_compatible(m):
                if debug: print("Invalid payload")
                t += 1

            if t >= self.threshold:
                if debug: print("Threshold reached")
                return 'Attack'

        return 'Normal'

def runGan(img, discriminator, opt = get_cmd_args(), cuda=True):
    # Cria um tensor de ponto flutuante
    FloatTensor = torch.FloatTensor

    # Configuração da entrada
    real_imgs = Variable(img.type(FloatTensor))

    # Passa as imagens reais pelo discriminador para obter a validade
    real_pred, real_aux = discriminator(real_imgs)

    # Calcula a acuracia do descriminador
    pred = real_aux.data.cpu().numpy()

    return np.argmax(pred, axis=1)[0]

class IDS:
    def __init__(self, filter : Filter, model : Discriminator, opt, window_size = 8, stride = 4):
        self.filter = filter
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.win = []
        self.opt = opt
        self.transform = get_transform(opt.img_size)
        self.correct = 0
        self.count = {
            NORMAL_MSG: [0, 0],
            DOS_MSG: [0, 0],
            FUZZY_MSG: [0, 0],
            FALS_MSG: [0, 0],
            IMP_MSG: [0, 0],
            'Filtered Attack': [0, 0]
        }

    def get_accuracy(self):
        for k, v in self.count.items():
            print(f"{k}:")
            print(f"\tCorrect: {v[0]}")
            print(f"\tIncorrect: {v[1]}")

    def test(self, msg : CANMessage):
        self.win.append(msg)
        if len(self.win) != self.window_size:
            return None

        label = 'Normal'

        # Pass the window to the filter
        result = self.filter.test(self.win)
        if result == 'Attack':
            label = 'Filtered Attack'
        elif result == 'Normal':
            # Run the GAN model
            img = self.transform(create_can_image(self.win, mirror=self.opt.mirror_img))
            # Turning the image into a batch of size 1
            img = torch.unsqueeze(img, 0)

            label = MSGS_TYPES_VALUES[ runGan(img, self.model, self.opt) ]

        is_attack = any(m.label != MSGS_TYPES[NORMAL_MSG] for m in self.win)
        # for m in self.win:
        #     print(str(m))
        # print(is_attack)
        # print(label)
        predicted_attack = label != 'Normal'
        if (not is_attack and not predicted_attack) or (is_attack and predicted_attack):
            # print(f"Predicted: {label}, Real: {'Attack' if is_attack else 'Normal'}")
            self.count[label][0] += 1
        else:
            self.count[label][1] += 1

        self.win = self.win[self.stride:]
