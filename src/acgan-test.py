import torch
from acgan import testGan, trainGan
from canDataset import CANDataset, DOS_MSG
from ganClasses import Discriminator
from utils import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

opt = get_cmd_args()
setseed(42)

if opt.test:
    model = Discriminator(opt)
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    dataset_dos = CANDataset([
        ('../attacks/validation/0-dos-candump-2024-07-10_184308.log', DOS_MSG),
    ], opt, transform=get_transform(opt.img_size))
    dataset_fal = CANDataset([
        ('../attacks/validation/1-falsifying-candump-2024-07-10_184439.log', FALS_MSG),
    ], opt, transform=get_transform(opt.img_size))
    dataset_fuz = CANDataset([
        ('../attacks/validation/2-fuzzing-candump-2024-07-10_184609.log', FUZZY_MSG),
    ], opt, transform=get_transform(opt.img_size))
    dataset_imp = CANDataset([
        ('../attacks/validation/3-impersonation-candump-2024-07-10_184739.log', IMP_MSG),
    ], opt, transform=get_transform(opt.img_size))
    dataset_general = CANDataset([
        ('../attacks/validation/3-impersonation-candump-2024-07-10_184739.log', IMP_MSG),
        ('../attacks/validation/0-dos-candump-2024-07-10_184308.log', DOS_MSG),
        ('../attacks/validation/1-falsifying-candump-2024-07-10_184439.log', FALS_MSG),
        ('../attacks/validation/2-fuzzing-candump-2024-07-10_184609.log', FUZZY_MSG),
    ], opt, transform=get_transform(opt.img_size))

    for type_of_attack, dataset in [(DOS_MSG, dataset_dos), (FALS_MSG, dataset_fal), (FUZZY_MSG, dataset_fuz), (IMP_MSG, dataset_imp), ('Every Attack', dataset_general)]:

        print('\nDataset:', type_of_attack)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
        testGan(dataloader, model, opt, cuda=False)
        print('')
else:
    trainGan([
        ('../attacks/validation/0-dos-candump-2024-07-10_184308.log', DOS_MSG),
        ('../attacks/validation/1-falsifying-candump-2024-07-10_184439.log', FALS_MSG),
        ('../attacks/validation/2-fuzzing-candump-2024-07-10_184609.log', FUZZY_MSG),
        ('../attacks/validation/3-impersonation-candump-2024-07-10_184739.log', IMP_MSG),
    ])
