import torch

from config import load_config
from model import generator
from PIL import Image
import numpy as np

args = load_config()
device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def denormalize(normalized_image):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(device)
    return normalized_image * std + mean


def trans_to_PIL(tensor:torch.Tensor):
    tensor = denormalize(tensor)
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    return Image.fromarray(img)


def ckpt2pth_G(ckpt_path):
    G = generator.Generator(args.feat_channel, args.img_channel).to(device)
    check_point = torch.load(ckpt_path)
    G.load_state_dict(check_point['G_state_dict'])
    torch.save(G.state_dict())