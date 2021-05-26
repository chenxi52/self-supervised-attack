import torch 
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from transform import overlap
from model import CPCModel
import tqdm
from prep_dataset import our_dataset
from torch.backends import cudnn
from resnet import resnet_v2_101 as resnet

parser=argparse.ArgumentParser(description='Train')
parser.add_argument('--n_imgs',type=int,default=20,help='number of reference images')
parser.add_argument('--n_iters',type=int,default=15000)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--mode',type=str,default='train')
parser.add_argument('--save_dir',type=str,default='./trained_model')
parser.add_argument('--gpu',type=int,help='gpu used')
parser.add_argument('--start',type=int,default=0)
parser.add_argument('--end',type=int,default=250)

def compute_acc(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return torch.sum(torch.eq(pred, labels)*1) / labels.size(0)



if __name__=='__main__':
    args=parser.parse_args()
    cudnn.benchmark=False#不做动态卷积优化，
    cudnn.deterministic = True#每次返回的卷积算法将是确定的，即默认算法.果配合上设置 Torch 的随机种子为固定值的话，
    # 应该可以保证每次运行网络的时候相同输入的输出是固定的。
    SEED=0
    torch.manual_seed(SEED)

    device = torch.device('cuda:{}'.format(args.gpu))
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()])
    img_num=args.n_imgs//2
    batch_size=img_num*2
    imageNet_dataset=our_dataset(data_dir='data/ILSVRC2012_img_val',data_csv='data/selected_data.csv',mode='train',
                                 img_num=img_num,transform=transform)
    loader = DataLoader(imageNet_dataset, batch_size=batch_size, shuffle=False)

    #scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.1, last_epoch=-1)

    l = []
    acc = []
    for i, (x, _) in enumerate(loader):#x[nImages,3,256,256]
        if not args.start<=i<args.end:
            continue
        model = CPCModel(3, 64, 3, 9).to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        for iter in range(args.n_iters):
            opt.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss, accuracy = model(x.to(device))
                loss.backward()
                opt.step()
            l.append(loss.item())
            acc.append(accuracy)
            if iter % 10 == 0 :
                print(f'loss {sum(l[-100:])/100}, accuracy {np.mean(acc[-100:])} at itertion {i} at epoch {iter}')
        #scheduler.step()
        model.eval()
        model.save_encoder(args.save_dir,i)