import torch
from PIL import Image
import numpy as np
from model import generator
from config import load_config

args = load_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def trans_to_PIL(tensor:torch.Tensor):
    img = tensor.detach().cpu().numpy()
    print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    return Image.fromarray(img)


def gen_by_vec(feature_vector):
    feature_vector = torch.tensor(feature_vector).to(device)
    feature_vector = feature_vector.unsqueeze(0)
    G = generator.Generator(args.noise_features).to(device)
    print(G.eval())
    G.load_state_dict(torch.load('./trained_model/G_1689860307.pth'))
    feature_vector = torch.tensor(feature_vector).to(device)

    pic = G(feature_vector).squeeze()
    return trans_to_PIL(pic)



if __name__ == '__main__':
    vec = [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0]
    pic = gen_by_vec(vec)
    print(pic)