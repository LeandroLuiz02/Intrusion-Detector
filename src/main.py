from utils import *
from canDataset import CANDataset
from torch.utils.data import DataLoader

opt = get_cmd_args()
setseed(42)

gan = Discriminator(opt)
gan.load_state_dict(torch.load('./model.pth'))
gan.eval()

ids = IDS(
    filter=Filter(
        CommunicationMatrix('./communication_matrix.json'),
        window_size=opt.window_size,
        threshold=2,
        tolerance=2e-4,
        enable_time=False
    ),
    model=gan,
    opt=opt)

dataset = CANDataset([
    ('../attacks/validation/0-dos-candump-2024-07-10_184308.log', DOS_MSG),
    ('../attacks/validation/1-falsifying-candump-2024-07-10_184439.log', FALS_MSG),
    ('../attacks/validation/2-fuzzing-candump-2024-07-10_184609.log', FUZZY_MSG),
    ('../attacks/validation/3-impersonation-candump-2024-07-10_184739.log', IMP_MSG),
], opt, transform=get_transform(opt.img_size))

for m in dataset.msgs:
    ids.test(m)

ids.get_accuracy()
