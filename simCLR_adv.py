import torch.optim
import torch
from Models.resnet_multi_bn import resnet18,proj_head
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from optimizer.lars import LARS
from prep_dataset import our_dataset
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import AverageMeter,nt_xent
import time
#import matplotlib.pyplot as plt

def PGD_contrastive(model, inputs, eps=8. / 255., alpha=2. / 255., iters=10, singleImg=False, feature_gene=None, sameBN=False):
    # init
    delta = torch.rand_like(inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)

    if singleImg:
        # project half of the delta to be zero
        idx = [i for i in range(1, delta.data.shape[0], 2)]
        delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    for i in range(iters):
        if feature_gene is None:
            if sameBN:
                features = model.eval()(inputs + delta, 'normal')
            else:
                features = model.eval()(inputs + delta, 'pgd')
        else:
            features = feature_gene(model, inputs + delta, 'eval')

        model.zero_grad()
        loss = nt_xent(features)
        #为什么feature自乘，feature标准化吗
        loss.backward()
        # print("loss is {}".format(loss))

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(inputs + delta.data, min=0, max=1) - inputs

        if singleImg:
            # project half of the delta to be zero
            idx = [i for i in range(1, delta.data.shape[0], 2)]
            delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    return (inputs + delta).detach()

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr

def train(train_loader, model, optimizer, scheduler, epoch):

    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()
    for i, (inputs) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()

        inputs_adv = PGD_contrastive(model, inputs, iters=5, singleImg=False)
        features_adv = model.train()(inputs_adv, 'pgd')
        features = model.train()(inputs, 'normal')
        loss = (nt_xent(features) + nt_xent(features_adv))/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(float(loss.detach().cpu()), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        # torch.cuda.empty_cache()
        if i % 5 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f}\t'
                     'iter_train_time: {train_time.avg:.2f}\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))
def main():
    device=torch.device('cuda:4')
    bn_names = ['normal','pgd' ]
    model=resnet18(pretrained=False,bn_names=bn_names)

    ch=model.fc.in_features
    model.fc=proj_head(ch,bn_names=bn_names,twoLayerProj=False)
    model.to(device)
    cudnn.benchmark=True

    strength=1.0
    rnd_color_jitter = transforms.RandomApply(
        [transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength)
    rnd_gray = transforms.RandomGrayscale(p=0.2 * strength)
    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(1.0 - 0.9 * strength, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
    ])
    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    n_imgs = 20
    img_num = n_imgs // 4
    batch_size = n_imgs

    ##################
    ds = our_dataset(data_dir='datafile/ILSVRC2012_img_val', data_csv='data/selected_data.csv', mode='train',
                     img_num=img_num, transform=tfs_train)
    da_ld = DataLoader(ds, batch_size=batch_size, shuffle=False)

    optimizer = LARS(model.parameters(), lr=5.0, weight_decay=1e-6)
    epochs=200
    scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    epochs * len(da_ld),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / 5.0,
                                                    warmup_steps=10 * len(da_ld))
        )
    for epoch in range(epochs):
        train(da_ld, model, optimizer, scheduler, epoch)
        torch.save({
            'epoch':epoch,
            'state_dict':model.state_dict(),
            'optim':optimizer.state_dict(),
        },'./trained_model/simCLR/model.pt')

if __name__=='__main__':
    main()
