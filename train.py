import time
import os

import torch

from config import load_config
from torch.optim import lr_scheduler
import torch.nn as nn

from model import discriminator, generator
from model.utils import init_net

import torchvision
torch.set_printoptions(profile="full", linewidth=100)
from icecream import ic


args = load_config()
train_id = int(time.time())
resume_epoch = 0
print(f'Training ID: {train_id}')


device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

# model
G = init_net(generator.Generator(args.noise_features), args.init_type, args.init_gain).to(device)
D = init_net(discriminator.Discriminator(args.channel), args.init_type, args.init_gain).to(device)

# retrained / continuous training
try:
    most_recent_check_point = os.listdir('checkpoint')[-1]
    ckpt_path = os.path.join('checkpoint', most_recent_check_point)
    check_point = torch.load(ckpt_path)
    # load model
    G.load_state_dict(check_point['G_state_dict'])
    D.load_state_dict(check_point['D_state_dict'])
    resume_epoch = check_point['epoch']
    print(f'Successfully load checkpoint {most_recent_check_point}, '
          f'start training from epoch {resume_epoch + 1}')
except:
    print('fail to load checkpoint, train from zero beginning')

# optimizer and schedular
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

lr_scheduler_G = lr_scheduler.SequentialLR(
    optimizer_G, schedulers=[lr_scheduler.ExponentialLR(optimizer_G, gamma=1),
                             lr_scheduler.CosineAnnealingWarmRestarts(
                                 optimizer_G,
                                 T_0=10, T_mult=2, eta_min=1e-5
                             )], milestones=[5]
)

lr_scheduler_D = lr_scheduler.SequentialLR(
    optimizer_D, schedulers=[lr_scheduler.ExponentialLR(optimizer_D, gamma=0.9),
                             lr_scheduler.CosineAnnealingWarmRestarts(
                                 optimizer_D,
                                 T_0=10, T_mult=2, eta_min=5e-6
                             )], milestones=[5]
)
for _ in range(resume_epoch):
    lr_scheduler_G.step()
    lr_scheduler_D.step()

# criterion
# Binary cross entropy and Pixel loss
criterion_GAN = nn.CrossEntropyLoss()
pixel_loss = nn.L1Loss()

print('***start training***')
# training
for epoch in range(resume_epoch + 1, args.epochs):
    epoch_start_time = time.time()
    print(f'{"*" * 20} Start epoch {epoch}/{args.epochs} {"*" * 20}')

    for i in range(args.batch_num):

        z = torch.randn((args.batch_size, args.noise_features, 1, 1)).to(device)

        # both of real images and fake images are generated
        # real images are generated from algorithm
        # fake images are generated from generator

        real_img = ""# versa work
        fake_img = G(z)

        # Generator back-prop
        optimizer_G.zero_grad()

        pred_fake = D(fake_img)
        target_real = (torch.ones(pred_fake.shape) * 0.95).to(device)
        target_fake = (torch.ones(pred_fake.shape) * 0.05).to(device)
        loss_G = criterion_GAN(pred_fake, target_real) + pixel_loss(fake_img, real_img)

        loss_G.backward()
        optimizer_G.step()

        # Discriminator back-prop
        optimizer_D.zero_grad()

        pred_real = D(real_img)
        loss_pred_real = criterion_GAN(pred_real, target_real)
        # ic(pred_real, loss_pred_real)
        pred_fake = D(fake_img.detach())
        loss_pred_fake = criterion_GAN(pred_fake, target_fake)
        # ic(pred_fake, loss_pred_fake)
        loss_D = (loss_pred_real + loss_pred_fake) * 0.5

        loss_D.backward()
        optimizer_D.step()

        # update log

        # show result
        # if i % args.print_interval == 0:
        #     print(f'epoch: {epoch}/{args.epochs}\tbatch: {i}/{len(data_loader)}\t'
        #           f'loss_G: {loss_G:0.6f}\tloss_D: {loss_D:0.6f}\t'
        #           f'|| learning rate_G: {optimizer_G.state_dict()["param_groups"][0]["lr"]:0.8f}\t'
        #           f'learning rate_D: {optimizer_D.state_dict()["param_groups"][0]["lr"]:0.8f}\t')
        #
        #     # ic(pred_fake, pred_real)
        #     # ic(pred_fake.shape)
        #     os.makedirs('output', exist_ok=True)
        #     fake_img = G(z)
        #     img = torch.cat([torch.cat([gen_img(args, G, device) for _ in range(4)], dim=-1) for _ in range(4)], dim=-2)
        #     torchvision.utils.save_image(img, f'output/{train_id}_{epoch}_{i}.jpg')

    # scheduler
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    print(f'End of epoch: {epoch}/{args.epochs}\t time taken: {time.time() - epoch_start_time:0.2f}')

    # save ckpt
    os.makedirs('checkpoint', exist_ok=True)
    torch.save({'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                }, f'checkpoint/{train_id}_{epoch:03d}.pt')

# save model
os.makedirs('trained_model', exist_ok=True)
try:
    torch.save(G.state_dict(), f'./trained_model/G_{train_id}.pth')
    print(f'Successfully saves the model ./trained_model/G_{train_id}.pth')
except:
    print('Fail to save the model, this project will automatically use the latest checkpoint to recover the model')