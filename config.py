from argparse import ArgumentParser


def load_config():
    args = ArgumentParser()

    args.add_argument('--dataset', type=str, default='./data', help='path of the dataset')
    args.add_argument('--img_channel', type=int, default=3, help='number of the image channel')
    args.add_argument('--feat_channel', type=int, default=14, help='number of the feature channel')

    args.add_argument('--init_type', type=str, default='normal', help='init function')

    args.add_argument('--init_gain', type=float, default=0.02, help='init gain')

    # device
    args.add_argument('--cuda', type=bool, default=True, help='use cuda training')

    # lr scheduler
    args.add_argument('--start_epoch', type=int, default=0, help='start epochs')
    args.add_argument('--decay_epoch', type=int, default=100, help='decay start epoch')

    # training
    args.add_argument('--batch_size', type=int, default=32, help='batch size number')
    args.add_argument('--alpha', type=float, default=0.2, help='pixel loss weight')
    args.add_argument('--lr_g', type=float, default=2e-4, help='learning rate of generator')
    args.add_argument('--lr_d', type=float, default=2e-4, help='learning rate of discriminator')
    args.add_argument('--epochs', type=int, default=200, help='number of epochs')
    args.add_argument('--output_channel', type=int, default=3, help='size of the output image channel')

    # print option
    args.add_argument('--print_interval', type=int, default=10, help='print intervel')

    # generation option
    args.add_argument('--model', type=str, default='G_1690361611.pth', help='name of the model you want to use')
    args.add_argument('--gen_number', type=int, default=10, help='number of image you want to generate')

    return args.parse_args()