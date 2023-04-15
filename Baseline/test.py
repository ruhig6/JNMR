import argparse

import torch
import numpy as np
import skimage.metrics
from torchvision.utils import save_image as imwrite
from torch.utils.data import DataLoader

from lpips_pytorch import lpips
from datasets import *
from models.main import AdaCoFNet
from utility import print_and_save, count_network_parameters
from collections import OrderedDict


def Vimeo90K_test(args, model, file_stream):

    val_dataset, test_dataset = Vimeo90K_interp_test(args.data_dir)
    if args.mode == 'val':
        print("start val")
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8)
    else:
        print("start test")
        val_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)
     

    transform = transforms.Compose([transforms.ToTensor()])

    img_out_dir = args.out_dir + '/vimeo90k'
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    av_psnr = 0
    av_ssim = 0
    av_lps = 0
    for batch_idx, (frame0, frame1, frame2, frame3, frame4) in enumerate(val_loader):
        frame0, frame1, frame3, frame4 = frame0.cuda(), frame1.cuda(), frame3.cuda(), frame4.cuda()
        
        output = model(frame0, frame1, frame3, frame4)
        
        imwrite(output, img_out_dir + '/' + str(batch_idx) + '.png', range=(-1,1))

        ref = transform(Image.open(img_out_dir + '/' + str(batch_idx) + '.png')).numpy()

        lps = lpips(frame2.cuda(), torch.tensor(ref).unsqueeze(0).cuda(), net_type='squeeze')
        psnr = skimage.metrics.peak_signal_noise_ratio(image_true=frame2.squeeze().numpy(), image_test=ref,
                                                       data_range=1)
        ssim = skimage.metrics.structural_similarity(np.transpose(frame2.squeeze().numpy(), (1, 2, 0)),
                                                     np.transpose(ref, (1, 2, 0)), data_range=1, multichannel=True)
        
        print('idx: %d, psnr: %f, ssim: %f, lpips: %f' % (batch_idx, psnr, ssim, lps.item()))

        av_psnr += psnr
        av_ssim += ssim
        av_lps += lps.item()

    av_psnr /= len(val_loader)
    av_ssim /= len(val_loader)
    av_lps /= len(val_loader)
    
    msg = '\n{:<15s}{:<20.16f}{:<23.16f}{:<23.16f}'.format('Average: ', av_psnr, av_ssim, av_lps)
    if file_stream:
        print_and_save(msg, file_stream)
    else:
        print(msg)

    return av_psnr


#########################################

def parse_args():
    parser = argparse.ArgumentParser(description='Compression-Driven Frame Interpolation Evaluation')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pth')
    parser.add_argument('--out_dir', type=str, default='./test/results')
    parser.add_argument('--data_dir', type=str, default='../../vimeo/')
    parser.add_argument('--log', default='./test/results/log.txt', help='the log file in training')
    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--dilation', type=int, default=2)
    parser.add_argument('--mode', type=str, default='test')

    return parser.parse_args()


def main():

    args = parse_args()
    
    logfile = open(args.log, 'w')
    
    torch.cuda.set_device(args.gpu_id)

    model = AdaCoFNet(args).cuda()
    print('===============================')
    print("# of model parameters is: " + str(count_network_parameters(model)))

    print('Loading the model...')
    
# load params

    checkpoint = torch.load(args.checkpoint)
    check = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in check.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.eval()

    
    print('===============================')
    print('Test: Vimeo-90K')
    _ = Vimeo90K_test(args, model, file_stream=logfile)
    
    logfile.close()
    

if __name__ == "__main__":
    
    main()


