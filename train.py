import argparse
import itertools
from copy import deepcopy
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath("/content"))
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = os.path.dirname(os.path.abspath("/content/code"))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(".")

from models_style import Generator
from models_style import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

params = argparse.Namespace()
params.dataset = "facades"
params.num_epochs = 30
params.batch_size = 1
params.lr = 0.0002
params.decay_epoch = 28
params.input_size = 256
params.resize_scale =286
params.crop_size = 256
params.input_nc = 3
params.output_nc = 3
params.dataroot = os.path.join(os.path.abspath('.'), 'datasets/horse2zebra/')
params.device = "cuda" if torch.cuda.is_available() else "cpu"

netG_A2B = Generator(params.input_nc, params.output_nc).to(params.device)
netG_B2A = Generator(params.output_nc, params.input_nc).to(params.device)
netD_A = Discriminator(params.input_nc).to(params.device)
netD_B = Discriminator(params.output_nc).to(params.device)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=params.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=params.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=params.lr, betas=(0.5, 0.999))

lr_sched_params = LambdaLR(params.num_epochs, 0, params.decay_epoch).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=deepcopy(lr_sched_params))
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=deepcopy(lr_sched_params))
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=deepcopy(lr_sched_params))

input_A = torch.Tensor(params.batch_size, params.input_nc, params.crop_size, params.crop_size)
input_B = torch.Tensor(params.batch_size, params.output_nc, params.crop_size, params.crop_size)
target_real =torch.tensor(params.batch_size, requires_grad=False, dtype=torch.float).fill_(1.0).to(params.device)
target_fake = torch.tensor(params.batch_size, requires_grad=False, dtype=torch.float).fill_(0.0).to(params.device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

transforms_ = [ transforms.Resize(int(params.resize_scale), Image.BICUBIC),
                transforms.RandomCrop(params.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(params.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=params.batch_size,
                        shuffle=True,
                        num_workers=4)

logger = Logger(params.num_epochs, len(dataloader))


###################################

###### Training ######
for epoch in range(0, params.num_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch['A'].to(params.device, torch.float)
        real_B = batch['B'].to(params.device, torch.float)

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')
