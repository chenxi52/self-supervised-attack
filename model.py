import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

from itertools import combinations_with_replacement


def image_preprocess(x):
    arr = []
    for i in range(7):
        for j in range(7):
            arr.append(x[:, :, i * 32:i * 32 + 64, j * 32:j * 32 + 64])  # batch,crop,crop,channel
    x = torch.cat(arr, 0)  # 把list里面的tensor按0维合并,先按一行中的patch组合，再是列
    return x
class encoder(nn.Module):
    def __init__(self, in_channels, dim, n_resblocks):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 5, 2, 2), nn.ReLU()]#in_channel,out_channel,kernel_size,stride,paddding
        for _ in range(n_resblocks):
            layers += [Resblock(dim)]
        layers.append(nn.AdaptiveAvgPool2d(1))#均值池化， 输出特征图大小和 input planes大小相等.
        self.encoder = nn.Sequential(*layers)
    def forward(self, x):#x_size:(49*20,64,64,3)
        zs = []
        for i in range(x.size(0)):#
            x_input=torch.unsqueeze(x,0)
            cc=x_input[:,i,:,:,:]
            zs.append(self.encoder(x_input[:,i,:,:,:]))
        zs = torch.stack(zs, dim=0).squeeze()
        return zs #(7*7*20,dimension)

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

class CPCModel(nn.Module): 
    def __init__(self, in_channels, dim, n_resblocks, n_time_steps):
        super().__init__()
        
        self.encoder = encoder(in_channels, dim, n_resblocks)

        self.gru = nn.GRU(dim, 256)#时间序列预测
        self.WK = nn.ModuleList([nn.Linear(256, dim) for _ in range(n_time_steps)])
        self.logsoftmax = nn.LogSoftmax(dim=0)


    def forward(self, x):
        # x size : batch_size x 3 x 64 x 64
        #imageNet
        b = x.size(0)
        x_in=image_preprocess(x)
        zs = self.encoder(x_in) # e.i batch_size x 9 x 256
        n=7
        zs=torch.reshape(zs,shape=[b,7,7,64])
        features=zs
        zs=zs.permute(0,2,1,3)#先行关系，再列关系
        tmp=[]
        for i in range(b):
            for j in range(7):
                tmp.append(zs[i][j])
        zs=torch.stack(tmp,0)
        batch_size=b*7#每一行是一个训练样例？
        # for random row
        nl = []
        nrr = []
        nrri = []
        K=2#要预测的z有两个？
        for i in range(K):#超参？
            nlist = torch.arange(0, n)
            nlist = nlist[nlist != (n - K + i)]#不等于5|6
            nl.append(nlist)
            nl[i] = nl[i][torch.randperm(len(nl[i]))]#打乱nl[i]
            nrr.append([nl[i] for j in range(batch_size)])#nrr[batch_size个[3，4，6，2，1，0]，batch_size个[2,5,1,3,4,0]]//随机
        nrri = [torch.stack([nrr[j][i][0] for j in range(K)], 0) for i in range(batch_size)]#例batch_size个[3,2]

        Y = []
        Y_label = np.zeros((batch_size), dtype=np.float32)
        n_p = batch_size // 2

        for i in range(batch_size):
            if i <= n_p:
                Y.append(torch.unsqueeze(features[int(i / n), -K:, i % n, :], 0))#最后的K个 feature size:(b,7,7,64)
                Y_label[i] = 1
            else:
                Y.append(torch.unsqueeze(torch.gather(features[int(i / n)],0, nrri[i]) [:, i % n, :], dim=0))

        Y = torch.cat(Y,dim=0)
        Y_label = torch.tensor(Y_label, dtype=torch.float32)

        nr = torch.tensor(list(range(batch_size)), dtype=torch.int32)[torch.randperm(batch_size)]

        ## cpc
        X_len = [5] * batch_size
        X_len = torch.tensor(X_len, dtype=torch.int32)


        t = 4 #torch.randint(1, 8, size=(1,)).long() # 1 to 7, 4
        zt, ztk = zs[:, :t, :], zs[:, t:, :] # 
        zt = zt.permute(1,0,2)
        out, ct = self.gru(zt) # b x 256 
        preds = []
        
        ct = ct.squeeze()
        
        for linear in self.WK[t:]:
            preds.append(linear(ct))
        preds = torch.stack(preds, dim=1) # b x 4 x 256
        total_loss = []
        accuracies = []
        for i in range(b):#对于batch每个样本计算
            p_b = preds.select(0,i)#该pred
            ftk = self.logsoftmax(torch.exp(torch.sum(torch.mul(ztk, p_b), dim=-1)).squeeze())
            m = torch.argmax(ftk, dim=0).detach()
            t = torch.tensor([i] * 5).cuda()
            
            acc = torch.sum(torch.eq(m,t))/5
            total_loss.append(ftk)
            accuracies.append(acc.cpu().item())

        total_loss = torch.stack(total_loss, dim=0)
        targets = [torch.ones(size=(5,)).long().cuda() * i for i in range(b)]
        targets = torch.stack(targets,dim=0)
        total_loss = F.nll_loss(total_loss, targets)
        return total_loss, np.mean(accuracies)



    def save_encoder(self,save_dir,iter_ind):
        torch.save(self.encoder.state_dict(), './{}/{}_encoder_weights.pt'.format(save_dir,iter_ind))
        print('encoder saved !')


if __name__ == '__main__':
    x = torch.Tensor(size=(3, 9, 1, 14, 14)).uniform_().cuda()
    cpc = CPCModel(1, 64, 1, 9).cuda()

    cpc(x)

    





