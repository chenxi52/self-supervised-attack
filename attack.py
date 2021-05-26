import os
import torch
import time
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import torchvision
import random
import argparse
import torch.nn.functional as F
import numpy as np
from torch.backends import cudnn
import sys
from prep_dataset import our_dataset
import csv
from model import CPCModel,encoder
from classifier import classifier

parser=argparse.ArgumentParser(description='attack')
parser.add_argument('--epsilon',type=float,default=0.1)
parser.add_argument('--ila_niters',type=int,default=100)
parser.add_argument('--ce_niters',type=int,default=200)
parser.add_argument('--save_dir',type=str,default='./adv_imgs')
parser.add_argument('--start',type=int,default=0)
parser.add_argument('--gpu',type=int)
args=parser.parse_args()
device=torch.device('cuda:{}'.format(args.gpu))

class ILA(torch.nn.Module):
    def __init__(self):
        super(ILA, self).__init__()

    def forward(self, ori_mid, tar_mid, att_mid):
        bs = ori_mid.shape[0]
        ori_mid = ori_mid.view(bs, -1)
        tar_mid = tar_mid.view(bs, -1)
        att_mid = att_mid.view(bs, -1)
        W = att_mid - ori_mid
        V = tar_mid - ori_mid
        V = V / V.norm(p=2, dim=1, keepdim=True)
        ILA = (W * V).sum() / bs
        return ILA

def save_attack_img(img, file_dir):
    T.ToPILImage()(img.data.cpu()).save(file_dir)

def initialize_model():
    model=encoder(1,64,3).to(device)#in_channels, dim, n_resblocks
    model.load_state_dict(torch.load('./encoder_weights.pt'))
    classi=classifier(576,10).to(device)
    classi.load_state_dict(torch.load('./classifier_weight.pt'))
    model=nn.Sequential(model,classi).to(device)
    return model

def attack_ce_regu(model,ori_img,attack_niters,eps,alpha,n_imgs,ce_method):
    model.eval()
    ori_img=ori_img.to(device)
    nChannels=3
    tar_img=[]
    for i in range(n_imgs):
        tar_img.append(ori_img[[i,n_imgs+i]])
    for i in range(n_imgs):
        tar_img.append(ori_img[[n_imgs + i, i]])
    tar_img = torch.cat(tar_img, dim=0)
    tar_img = tar_img.reshape(2 * n_imgs, 2, nChannels, 224, 224)
    img=ori_img.clone()
    for i in range(attack_niters):
        if ce_method=='ifgsm':
            img_x=img
        elif ce_method=='pgd':
            img_x=img+img.new(img.size()).uniform_(-eps,eps)
        img_x.requires_grad_(True)
        logits=model(img_x)