import argparse
import sys

# Define argumentos padrões

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=5, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=16, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--patience", type=int, default=5, help="patience for early stopping")
parser.add_argument("--tolerance", type=int, default=0.01, help="tolerance for early stopping")
parser.add_argument("--dataset_max_size", type=int, default=2e5, help="maximum number of samples for each dataset")
parser.add_argument("--window_size", type=int, default=8, help="maximum number of samples for each dataset")
parser.add_argument("--stride", type=int, default=4, help="maximum number of samples for each dataset")
parser.add_argument("--pregenerate_imgs", type=bool, default=False, help="If True, pregenerate images, to avoid generating them every time")
parser.add_argument("--mirror_img", type=bool, default=False, help="If True, mirror images, to increase information")
parser.add_argument("--test", type=bool, default=False, help="If True, test the model")
opt = parser.parse_args()
print(opt)
