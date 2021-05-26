import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from transform import overlap
from model import CPCModel
from tqdm import tqdm
from prep_dataset import our_dataset
import argparse
parser=argparse.ArgumentParser(description='Train')
parser.add_argument('--n_imgs',type=int,default=20,help='number of reference images')
parser.add_argument('--n_iter',type=int,default=15000)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--mode',type=str,default='train')
parser.add_argument('--save_dir',type=str,default='./trained_model')
parser.add_argument('--gpu',type=int,help='gpu used')
args=parser.parse_args()

class Resblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels, in_channels//2, 1), 
                             nn.ReLU(),
                             nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1),
                             nn.ReLU(),
                             nn.Conv2d(in_channels//2, in_channels, 1))
    
    def forward(self, x):
        residual = x
        x = self.main(x)
        return F.relu(x + residual)

def compute_acc(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return torch.sum(torch.eq(pred, labels)*1) / labels.size(0)

class cl(nn.Module):
    def __init__(self, in_channels, dim, n_resblocks):
        super().__init__()
        layers = [nn.Conv2d(1, 64, 5, 2, 2), nn.ReLU()]
        for _ in range(n_resblocks):
            layers += [Resblock(dim)]
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*layers)
    def forward(self, x):
        zs = []
        for i in range(x.size(1)):
            zs.append(self.encoder(x[:,i,:, :,:])) # e.i batch_size x 9 x 256
        zs = torch.stack(zs, dim=1).squeeze()
        return zs

class classifier(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(in_feat, in_feat//4),
                    nn.ReLU(),
                    nn.Linear(in_feat//4, out_feat))
    def forward(self, x):
        return self.main(x)

img_num=args.n_imgs//2
transform = transforms.Compose([transforms.ToTensor(), overlap(14)])
imageNet_dataset=our_dataset(data_dir='data/ILSVRC2012_img_val',data_csv='data/selected_data.csv',mode='train',
                                 img_num=img_num,transform=transform)
loader = DataLoader(imageNet_dataset, batch_size=args.n_imgs, shuffle=False,num_workers=1)

model = cl(1, 64, 3).to('cuda')
model.load_state_dict(torch.load('/gel/usr/maabi11/Desktop/CPC/encoder_weights.pt'))

classi = classifier(576,10).to('cuda')
opt_cls = torch.optim.SGD(classi.parameters(), lr=0.1, momentum=.9)
scheduler = torch.optim.lr_scheduler.StepLR(opt_cls, 20, gamma=0.1, last_epoch=-1)
criterion = torch.nn.CrossEntropyLoss()

epochs = 100
model.eval()
l = []
acc = []
for epoch in tqdm(range(epochs)):
    for i, (x, labels) in enumerate(loader):
        labels = labels.cuda()
        opt_cls.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            logits = classi(model(x.cuda()).view(x.size(0), -1).detach())
            loss2 = criterion(logits, labels)
            loss2.backward()
            opt_cls.step()
            accuracy = compute_acc(logits.detach(), labels)
        l.append(loss2.item())
        acc.append(accuracy.cpu().item())
        if i % 100 == 0 :
            print(f'loss {sum(l[-100:])/100}, accuracy {np.mean(acc[-100:])} at itertion {i} at epoch {epoch}')
    scheduler.step()
torch.save(classi.state_dict(),'classifier_weight.pt')
print("classifier weight saved!")
