import os
import torch
from torchvision.models import resnet50
from prep_dataset import our_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import numpy as np
import random
from torchvision import utils as vutils
from torchvision.datasets import ImageFolder
from selfsupervised import remove_batchnorm,Identity
from torchvision.models import inception_v3,resnet152,densenet161,vgg19_bn,wide_resnet50_2,mobilenet_v2
import cv2
from scipy.stats import wasserstein_distance
import argparse
from advertorch.utils import NormalizeByChannelMeanStd
import torch.nn as nn

parser=argparse.ArgumentParser()
parser.add_argument('--mode',type=str,default='mae',choices=['mse','mae','cos'],help='the feature loss type')
parser.add_argument('--gpu',type=int,help='the gpu used')
parser.add_argument('--model',type=str)
parser.add_argument('--iters',type=int,default=1)
parser.add_argument('--weight_decay',type=float,default=0.0)
parser.add_argument('--momentum',type=float,default=0.9)

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True
CHECK = 1e-5
# CHECK = 1e-3
SAT_MIN = 0.5
MAX_EPS=0.08
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.count+=n
        self.sum+=val*n
        self.avg=self.sum/self.count
class MI_SGD(Optimizer):
    r'''Implemrnt stochastic gradient decent(optionally with momentum'''
    def __init__(self,params,lr=0,momentum=0,weight_decay=0,max_eps=10/255):
        defaults=dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            sign=True
        )
        super(MI_SGD,self).__init__(params,defaults)
        self.max_eps=max_eps
        self.sat = 0
        self.sat_prev = 0
    def __setstate__(self, state):
        super(MI_SGD, self).__setstate__(state)

    def rescale(self):
        for group in self.param_groups:
            if not group['sign']:
                continue
            for p in group['params']:
                self.sat_prev=self.sat
                self.sat=(p.data.abs()>=self.max_eps).sum().item()/p.data.numel()
                sat_change=abs(self.sat-self.sat_prev)
                if sat_change<CHECK and self.sat>SAT_MIN:#变化量少并且饱和度高,即大于最大eps的数量多
                    print('rescaled')
                    p.data=p.data/2

    def step(self,closure=None):
        loss=None
        for group in self.param_groups:
            weight_decay=group['weight_decay']
            momentum=group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p=p.grad.data
                if group['sign']:
                    d_p=d_p/(d_p.norm(1)+1e-12)
                if weight_decay!=0:
                    d_p.add_(weight_decay,p.data)
                if momentum!=0:
                    param_state=self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf=param_state['momentum_buffer']=torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf=param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p=buf
                if group['sign']:
                    p.data.add_(d_p.sign(),alpha=-group['lr'])
                    p.data=torch.clamp(p.data,-self.max_eps,self.max_eps)
                else:
                    p.data.add_(d_p,alpha=-group['lr'])
        return loss


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(val_loader, model, print_freq,noise=None,rand_noise=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    per_top1=AverageMeter()
    per_top5=AverageMeter()
    rand_top1=AverageMeter()
    rand_top5=AverageMeter()
    # switch to evaluate mode
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)
        per_input=torch.clamp(input+noise,0,1)
        rand_input=torch.clamp(input+rand_noise,0,1)
        with torch.no_grad():
            # compute output
            output = model(input)
            per_output=model(per_input)
            rand_output=model(rand_input)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            per_prec1,per_prec5=accuracy(per_output.data,target,topk=(1,5))
            rand_perc1,rand_perc5=accuracy(rand_output,target,topk=(1,5))
            top1.update(prec1[0])
            top5.update(prec5[0])
            per_top1.update(per_prec1[0])
            per_top5.update(per_prec5[0])
            rand_top1.update(rand_perc1[0])
            rand_top5.update(rand_perc5[0])
            # measure elapsed time
            if (i+1) % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Per_Prec@1 {per_top1.val:.3f} ({per_top1.avg:.3f})\t'
                      'Rand_Prec@1 {rand_top1.val:.3f} ({rand_top1.avg:.3f})\t'
                      'Prec@5 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Per_Prec@5 {per_top5.val:.3f} ({per_top5.avg:.3f})\t'
                      'Rand_Prec@5 {rand_top5.val:.3f} ({rand_top5.avg:.3f})\t'.format(
                    i+1, len(val_loader), loss=losses,
                    top1=top1, top5=top5,per_top1=per_top1,per_top5=per_top5,rand_top1=rand_top1,rand_top5=rand_top5))
   # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\t'.format(top1=top1, top5=top5,per_top1=per_top1,per_top5=per_top5,rand_top1=rand_top1))
def gen_noise(noise,loss_type):
    losses=AverageMeter()
    encode.eval()
    current_noise = noise
    optimizer = MI_SGD([{'params': [current_noise], 'lr': MAX_EPS/20, 'momentum': args.momentum, 'sign': True,'weight_decay':args.weight_decay}],
                       max_eps=MAX_EPS)
    print(optimizer)
    optimizer.zero_grad()

    losses=AverageMeter()

    for i,(imgs,label) in enumerate(da_ld):
        imgs=imgs.repeat(args.iters,1,1,1)
        for img in imgs:
            img=img.unsqueeze(0)
            encode.zero_grad()#每次要梯度置0
            img=img.to(device)
            with torch.no_grad():
                feature = encode(img)  # B,2048
            optimizer.zero_grad()

            perturbed_input=torch.clamp(img+current_noise,0,1)
            #归一化？
            perturbed_feature=encode(perturbed_input)
            if loss_type == 'mse':
                loss=torch.nn.MSELoss(feature,perturbed_feature)
            elif loss_type=='cos':
                loss=torch.cosine_similarity(feature,perturbed_feature,dim=1)
            #有误，两个feature之间的Wass距离是MAE(mean absolute err)
            elif loss_type=='mae':
                #loss=wasserstein_distance(feature.detach().cpu().squeeze(),perturbed_feature.detach().cpu().squeeze())
                loss=torch.sum(feature-perturbed_feature,dim=1)/feature.size(1)
            loss.backward()
            losses.update(loss.item())
            optimizer.step()
        break
    current_noise.requires_grad=False
    return torch.clamp(current_noise,-MAX_EPS,MAX_EPS)

if __name__=='__main__':
    args=parser.parse_args()
    gpu=args.gpu
    device=torch.device("cuda:{}".format(gpu))
    crop_size = 224
    encode = resnet50().to(device)
    encode.fc = Identity()
    remove_batchnorm(encode)
    state = torch.load('./trained_model/0_loss0.38962894678115845_encoder_weights.pt',
                       map_location=lambda storage, loc: storage.cuda(gpu))
    encode.load_state_dict(state)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])
    n_imgs = 20
    img_num = n_imgs // 4
    batch_size = n_imgs
    ds = our_dataset(data_dir='data/ILSVRC2012_img_val', data_csv='data/selected_data.csv', mode='train',
                     img_num=img_num, transform=data_transform)
    da_ld = DataLoader(ds, batch_size=batch_size, shuffle=False)

    noise = torch.FloatTensor(1, 3, crop_size, crop_size).uniform_(-MAX_EPS, MAX_EPS).to(device)
    noise.requires_grad = True
#loss_type:cos\mse\mae
    uni_noise=gen_noise(noise,loss_type='mae')
    uni_noise_ima=uni_noise.squeeze()
    vutils.save_image(1-uni_noise_ima,'./imgs/noise.png')
    vutils.save_image(uni_noise_ima,'./imgs/ori_noise.png')
    batch_size=100
    imageNet_ds=ImageFolder(root='./data/ILSVRC2012_img_val',transform=data_transform)
    test_dl=DataLoader(dataset=imageNet_ds,batch_size=batch_size,shuffle=False,num_workers=1)

    # imageNet_ds = our_dataset(data_dir='data/ILSVRC2012_img_val', data_csv='data/selected_data.csv', mode='train',
    #                  img_num=10, transform=data_transform)
    # test_dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    normalize=NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #model=inception_v3(pretrained=True).to(device).eval()

    if args.model=='vgg19_bn':
        model=vgg19_bn(pretrained=True).eval()
    elif args.model=='inception_v3':
        model=inception_v3(pretrained=True).eval()
    elif args.model=='resnet152':
        model=resnet152(pretrained=True).eval()
    elif args.model=='densenet161':
        model=densenet161(pretrained=True).eval()
    elif args.model=='wide_resnet50':
        model=wide_resnet50_2(pretrained=True).eval()
    elif args.model=='mobilenet_v2':
        model=mobilenet_v2(pretrained=True).eval()
    model=nn.Sequential(normalize,model).to(device)

    inc_acc=AverageMeter()
    def noise_batch(noise,batch_size):
        noise_batch=noise
        for i in range(batch_size-1):
            noise_batch=torch.cat((noise_batch,noise),dim=0)
        return noise_batch
    rand_noise=torch.FloatTensor(1,3,crop_size,crop_size).uniform_(-MAX_EPS,MAX_EPS).to(device)

    batch_noise=noise_batch(uni_noise,batch_size)
    rand_noise=noise_batch(rand_noise,batch_size)

    validate(test_dl,model,print_freq=10,noise=batch_noise,rand_noise=rand_noise)

