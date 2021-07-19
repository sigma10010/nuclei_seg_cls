import argparse
import os

from datasets.data_split import divide_data

import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from skimage import io
import os
import openslide as ops
from PIL import Image
from metrics.evaluation import *

from models.u_net import U_Net, AttU_Net, XXU_Net
from models.unet import UNet, UNet_V1, UNet_V2, UNet_V3, UNet_V4 # new version

import torch
import csv

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def wsi_prediction(s, model, n_class=2, patch_size = 400, step = 200, t=transforms.ToTensor()):
    
    width, height = s.size
    
    step = patch_size//2
    
    whole_size_i = height # y
    whole_size_j = width # x

    patch_size_i = patch_size
    patch_size_j = patch_size

    result = np.zeros((n_class,whole_size_i,whole_size_j))
    for i in range(0, whole_size_i, step):
        for j in range(0, whole_size_j, step):
            if i+patch_size_i>whole_size_i:
                i = whole_size_i-patch_size_i
            if j+patch_size_j>whole_size_j:
                j = whole_size_j-patch_size_j

            img = s.crop((j, i, j+patch_size, i+patch_size))
            img = t(img)[0:3,:,:].unsqueeze(0)
            SR, CR = model(img)
            SR = SR.squeeze().detach().numpy()

            if i==0 and j==0:
                result[:,i:i+patch_size_i,j:j+patch_size_j] = SR
            elif i==0 and j>0:
                overlay1 = result[:,i:i+patch_size_i,j:j+patch_size_j-step]
                overlay2 = SR[:,:,0:patch_size_j-step]
                SR[:,:,0:patch_size_j-step] = (overlay1+overlay2)/2
                result[:,i:i+patch_size_i,j:j+patch_size_j] = SR
            elif j==0 and i>0:
                overlay1 = result[:,i:i+patch_size_i-step,j:j+patch_size_j]
                overlay2 = SR[:,0:patch_size_j-step,:]
                SR[:,0:patch_size_j-step,:] = (overlay1+overlay2)/2
                result[:,i:i+patch_size_i,j:j+patch_size_j] = SR
            else:
                overlay1 = result[:,i:i+patch_size_i-step,j:j+patch_size_j]
                overlay2 = SR[:,0:patch_size_j-step,:]
                SR[:,0:patch_size_j-step,:] = (overlay1+overlay2)/2

                overlay3 = result[:,i+patch_size_i-step:i+patch_size_i,j:j+patch_size_j-step]
                overlay4 = SR[:,patch_size_i-step:patch_size_i,0:patch_size_j-step]
                SR[:,patch_size_i-step:patch_size_i,0:patch_size_j-step] = (overlay3+overlay4)/2

                result[:,i:i+patch_size_i,j:j+patch_size_j] = SR
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument('--depth', type=int, default=5, help='#conv blocks, 400-200-100-50-25')
    parser.add_argument('--width', type=int, default=32, help='#channel, 32-64-128-256-512')
    parser.add_argument('--image_size', type=int, default=400)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--category', type=int, default=1, help='category for evaluation label')
    parser.add_argument('--t', type=int, default=2, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--reduction_ratio', type=int, default=None, help='reduction ratio for attention layer') 
    parser.add_argument('--n_skip', type=int, default=4, help='number of skip-connection layers, <= depth-1') 
    parser.add_argument('--n_head', type=int, default=1, help='number of heads for prediction, 1 <= depth-1') 
    parser.add_argument('--att_mode', type=str, default='cbam', help='cbam/bam/se') 
    parser.add_argument('--conv_type', type=str, default='basic', help='basic/sk')
    parser.add_argument('--is_shortcut', type=str2bool, default=False)
    
    # dataset
    parser.add_argument('--name_dataset', type=str, default='monuseg', help='paip/monuseg')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.5)
    parser.add_argument('--start_epoch', type=int, default=0)
    
    # log
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=2)
    
    # loss
    parser.add_argument('--loss_type', type=str, default='nll', help='l1/nll/focal/iou/dice/multitask/nll+iou/nll+ssim/...')
    parser.add_argument('--alpha', type=float, default=1)        # alpha for l1 loss
    parser.add_argument('--gamma', type=float, default=1)        # gamma for l1/Focal loss
    parser.add_argument('--balance', type=list, default=None)   # balance factor

    # misc
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model_type', type=str, default='U_Net', help='XXU_Net/U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='/mnt/DATA_OTHER/moNuSeg/results/seg/')
    parser.add_argument('--wsi_path', type=str, default='/mnt/DATA_OTHER/moNuSeg/original_testing/test_vis/')
#     parser.add_argument('--wsi_path', type=str, default='/mnt/DATA_OTHER/moNuSeg/original_testing/tissue_Images/')
    parser.add_argument('--train_path', type=str, default='/mnt/DATA_OTHER/moNuSeg/original_training/patches/s400/all/')
    parser.add_argument('--train_anno_path', type=str, default=None)
    parser.add_argument('--valid_path', type=str, default='/mnt/DATA_OTHER/moNuSeg/original_testing/patches/s400/all/')
    parser.add_argument('--valid_anno_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default='./fundus_images/test/')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--fold', type=int, default=1, help='5-fold cross validation')
    parser.add_argument('--level', type=int, default=2, help='1/2')
    
    # other
    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    if config.n_skip>config.depth-1:
        config.n_skip = config.depth-1
    
    
    """Build model"""
    unet = None
    if config.model_type =='U_Net':
        unet = U_Net(img_ch=3, n_classes = config.n_classes, activation = torch.nn.Softmax(dim=1))
    elif config.model_type =='R2U_Net':
        unet = R2U_Net(img_ch=3,t=config.t, n_classes = config.n_classes, activation = torch.nn.Softmax(dim=1))
    elif config.model_type =='AttU_Net':
        unet = AttU_Net(img_ch=3, n_classes = config.n_classes, activation = torch.nn.Softmax(dim=1))
    elif config.model_type == 'R2AttU_Net':
        unet = R2AttU_Net(img_ch=3,t=config.t, n_classes = config.n_classes, activation = torch.nn.Softmax(dim=1))
    elif config.model_type in ['UNet', 'SEU_Net', 'CBAMU_Net', 'BAMU_Net']:
        unet = UNet(img_ch=3, n_classes=config.n_classes, init_features=config.width, network_depth=config.depth, reduction_ratio=config.reduction_ratio, att_mode = config.att_mode, activation = torch.nn.Softmax(dim=1))
    elif config.model_type in ['SKU_Net', 'SK-SC-U_Net', 'SK-SE-U_Net']:
        unet = UNet_V1(reduction_ratio=config.reduction_ratio, att_mode = config.att_mode, is_shortcut = config.is_shortcut, conv_type = config.conv_type, activation = torch.nn.Softmax(dim=1))
    elif config.model_type == 'MHU_Net':
        unet = UNet_V2(img_ch=3, n_classes=config.n_classes, n_head = config.n_head, is_head_selective = False, is_shortcut = False, activation = torch.nn.Softmax(dim=1))
    elif config.model_type == 'SCU_Net':
        unet = UNet_V3(img_ch=3, n_classes=config.n_classes, n_head = config.n_head, is_scale_selective = False, is_shortcut = True, activation = torch.nn.Softmax(dim=1))
    elif config.model_type in ['SSU_Net', 'SK-SSU_Net', 'SE-SSU_Net', 'SC-SSU_Net' ]:
        unet = UNet_V4(reduction_ratio=config.reduction_ratio, n_head = config.n_head, att_mode = config.att_mode, is_scale_selective = True, is_shortcut = config.is_shortcut, conv_type = config.conv_type, activation = torch.nn.Softmax(dim=1))
    else:
        raise NotImplementedError(config.model_type+" is not implemented")

        
    unet_path = os.path.join(config.model_path, '%s-%s-level%s-size%s-depth%s-width%s-n_classes%s-alpha%s-gamma%s-nhead%s-fold%s.pkl'%(config.model_type, config.loss_type, config.level, config.image_size, config.depth, config.width, config.n_classes, config.alpha, config.gamma, config.n_head, config.fold))
    print('try to load weights from: %s'%unet_path)
    unet.load_state_dict(torch.load(unet_path))
    unet.train(False)
    unet.eval()
    
    metrics = [get_MultiClassAccumulatedJSMetric(), get_MultiClassAccumulatedDCMetric()]
    
    imfs = []
    for imf in os.listdir(config.wsi_path):
        if imf.endswith(('.png','.jpg', '.jpeg', '.tif')):
            imfs.append(imf)
            
    jss = []
    t=transforms.ToTensor()
    for i, f in enumerate(imfs):
        print('predict img: %d/%d'%(i+1,len(imfs)))
        maskf = f.split('.')[0]+'_mask.bmp'
        
        im= Image.open(os.path.join(config.wsi_path, f))
        mask= Image.open(os.path.join(config.wsi_path, maskf))
        
        SR = wsi_prediction(im, unet)
        # save prediction as PIL
        segmap = SR.argmax(0).astype('uint8')
        segmap_pil = Image.fromarray((segmap*255), 'L')
        segmapf = f.split('.')[0]+ '_' + config.model_type + config.loss_type + '.bmp'
        segmap_pil.save(os.path.join(config.wsi_path, segmapf))
        
        tep_js = get_MultiClassJS(SR,t(mask),label=1)
        jss.append(tep_js)
        
        for metric in metrics:
            metric(SR, t(mask), label = 1)
        
        
    f = open(os.path.join(config.result_path,'result_monuseg.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow([config.model_type, config.loss_type, config.level, config.image_size, config.depth, config.width, config.n_classes, config.alpha, config.gamma, config.n_head, config.fold]+[float(metric.value()) for metric in metrics]+[js for js in jss])
    f.close()