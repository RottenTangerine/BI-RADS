import torch
from model import generator
from config import load_config
from utils import trans_to_PIL

args = load_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gen_by_vec(feature_vector):
    feature_vector = torch.tensor(feature_vector).to(device)
    feature_vector = feature_vector.view(1, -1, 1, 1)
    G = generator.Generator(args.feat_channel, args.img_channel).to(device)
    G.load_state_dict(torch.load(f'./trained_model/{args.model}'))

    pic = G(feature_vector).squeeze()
    return trans_to_PIL(pic)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    vec = [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0]
    pic = gen_by_vec(vec)
    plt.imshow(pic)
    plt.show()