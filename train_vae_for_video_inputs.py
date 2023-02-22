import argparse
import torch
import numpy as np
from torch import nn,optim
from videovae import VAEs, VideoData, VideoDataset
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader  #Gives easier data management by creating mini batches etc.
from tqdm import tqdm

def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))

def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    # The parser setting
    parser = argparse.ArgumentParser()
    parser = VAEs.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='/home/xu/XU-ishiguroLab/VAE/VAE_for_prediction/datasets/ucf101')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--LR_RATE', type=int, default=1e-5)
    parser.add_argument('--NUM_EPOCHS', type=int, default=3)
    args = parser.parse_args()
    # Load the Dataset
    Dataset = VideoDataset
    dataset = Dataset(args.data_path, args.sequence_length, train=True, resolution=args.resolution)
    # pre-make relevant cached files if necessary
    if dist.is_initialized():
        sampler = data.distributed.DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
    else:
        sampler = None
    # dataloader for the trainning
    dataloader_train = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None
    )
    #dataloader for the evaluation
    dataset = Dataset(args.data_path, args.sequence_length, train=False, resolution=args.resolution)
    dataloader_val = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None
    )

    # data = VideoData(args)
    # dataloader_train = data.train_dataloader()
    # data.test_dataloader()

    # Load the Model and set optimizer
    model = VAEs(args).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.LR_RATE)
    loss_fn = nn.MSELoss()
    #start Training
    for epoch in range(args.NUM_EPOCHS):
        losses = []
        KL_losses = []
        recon_losses = []
        model.train()
        loop = tqdm(dataloader_train)
        for x in loop:
            loop.set_description(f"Epoch {epoch}")
            x,_ = dict.values(x)
            x = x.to(DEVICE)
            x_reconstructed, mu, sigma = model.forward(x)
            # compute loss
            recon_loss = F.mse_loss(x_reconstructed, x) / 0.06
            kl_div = - 0.5 * torch.mean(torch.sum(1 + torch_log(sigma**2) - mu**2 - sigma**2, dim=1))
            loss = recon_loss + kl_div

            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            KL_losses.append(kl_div.cpu().detach().numpy())
            recon_losses.append(recon_loss.cpu().detach().numpy())
            loop.set_postfix(loss=loss.item(), KL_loss=kl_div.item(), Recon_loss=recon_loss.item() )
            # print(loss.item())

        losses_val = []
        model.eval()
        loop_val = tqdm(dataloader_val)
        for x in loop_val:
            loop.set_description(f"Eval_Epoch {epoch}")
            x, _ = dict.values(x)
            x = x.to(DEVICE)
            x_reconstructed, mu, sigma = model.forward(x)
            # compute loss
            loss_fn = nn.MSELoss()
            recon_loss = loss_fn(x, x_reconstructed)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = recon_loss + kl_div
            losses_val.append(loss.cpu().detach().numpy())
        print('EPOCH: %d    Train Lower Bound: %lf (KL_loss: %lf. reconstruction_loss: %lf)    Valid Lower Bound: %lf' %
              (epoch + 1, np.average(losses), np.average(KL_losses), np.average(recon_losses),
               np.average(losses_val)))
        # Save a checkpoint
        checkpoint_path = '/home/xu/XU-ishiguroLab/VAE/VAE_for_prediction'
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': loss,
                            'optimizer': optimizer.state_dict(),'alpha': loss.alpha, 'gamma': loss.gamma},
                           checkpoint_path + '/m-'  + '-' + str("%.4f" % loss) + '.pth.tar')
        # model.load_state_dict(torch.load('model.pth\pkl\pt')  # 一般形式为model_dict=model.load_state_dict(torch.load(PATH))

if __name__ == '__main__':
    main()