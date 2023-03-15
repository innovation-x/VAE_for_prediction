import argparse
from torchvision.io import read_video, write_video
from videovae import VAEs, VideoData, VideoDataset
import torch
import os.path
from torch import nn
from videovae.data import preprocess

if __name__ == '__main__':
    #get the abs_path of current project
    device = torch.device('cuda')
    abs_file = os.path.abspath(__file__)
    abs_dir = abs_file[:abs_file.rfind('\\')] if os.name == 'nt' else abs_file[:abs_file.rfind(r'/')]
    # path of video and checkpoints
    video_filename = os.path.join(abs_dir, 'datasets/mobileRobo/test/cafeteria_vid/1.mp4')
    checkpoint_path = os.path.join(abs_dir, 'checkpoints')
    sequence_length = 16
    resolution = 64
    # The parser setting
    parser = argparse.ArgumentParser()
    parser = VAEs.add_model_specific_args(parser)
    args = parser.parse_args()
    # load the VAE pretrained model (DataParallel)
    vae = VAEs(args).cuda()
    vae = nn.DataParallel(vae)
    vae.load_state_dict(torch.load(checkpoint_path + '/m495641.pth.tar'))
    vae = vae.module.eval()
    # read and preprocess the video
    video = read_video(video_filename, pts_unit='sec')[0]
    video = preprocess(video, resolution, sequence_length).unsqueeze(0).to(device)

    # reconstruct the video
    # mu, sigma = vae.encode(video)
    # epsilon = torch.randn_like(sigma)
    # z_reparametrized = mu + sigma * epsilon
    #
    z_reparametrized = torch.randn(1, args.z_dim).to(device)
    video_recon = vae.decode(z_reparametrized)
    B, C, T, H, W = video_recon.shape
    video_recon = video_recon.view(B*C,T,H,W)
    # Convert from float32 to uint8, use global minimum and global maximum - this is not the best solution
    min_val = video_recon.min()
    max_val = video_recon.max()
    video_recon_as_uint8 = ((video_recon - min_val) * 255 / (max_val - min_val)).to(torch.uint8)
    vid_arr = torch.permute(video_recon_as_uint8, (1, 2, 3, 0))  # Reorder the axes to be ordered as [T, H, W, C]
    vid_arr = vid_arr.cpu()
    write_video('demo.mp4', vid_arr, fps=10)

    video= video.view(B*C,T,H,W)
    min_val = video.min()
    max_val = video.max()
    video_as_uint8 = ((video - min_val) * 255 / (max_val - min_val)).to(torch.uint8)
    vid_arr1 = torch.permute(video_as_uint8, (1, 2, 3, 0))  # Reorder the axes to be ordered as [T, H, W, C]
    vid_arr1 = vid_arr1.cpu()
    write_video('origin.mp4', vid_arr1, fps=10)
