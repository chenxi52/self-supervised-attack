import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import torch.optim as optim
from torchvision.models import resnet18, resnet34
import torch.nn.functional as F
from resnet import ResNet_22
from prep_dataset import our_dataset
import pdb
import torchvision

class PixelCNN(nn.Module):
    def __init__(self, latent_dim):
        super(PixelCNN, self).__init__()
        # Conv2d: (input_channels, output_channels, kernel_size, padding)
        self.relu = nn.ReLU()

        self.model = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, (1, 1)),
            nn.ReLU(),
            nn.ConstantPad2d((1, 1, 0, 0), 0),
            nn.Conv2d(latent_dim, latent_dim, (1, 3)),
            nn.ConstantPad2d((0, 0, 0, 1), 0),
            nn.Conv2d(latent_dim, latent_dim, (2, 1)),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, (1, 1))
        )

    def forward(self, latents):
        # latents: [B, C, H, W]
        cres = latents

        for _ in range(5):
            c = self.model(cres)
            cres = cres + c
        cres = self.relu(cres)
        return cres


class CPC_loss(nn.Module):

    def __init__(self):
        super(CPC_loss, self).__init__()
        self.pixel_cnn = PixelCNN(2048)
        self.target_to_32 = nn.Conv2d(2048, 7, kernel_size=(1, 1))

        self.conv_1 = nn.Conv2d(2048, 7, kernel_size=(1, 1))
        self.conv_2 = nn.Conv2d(2048, 7, kernel_size=(1, 1))
        self.conv_3 = nn.Conv2d(2048, 7, kernel_size=(1, 1))
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, latents, device, target_dim=7, steps_to_ignore=2, steps_to_predict=3, emb_scale=0.1):
        # latents: [B, D, H, W]
        # aka:     [B, 512, 6, 6]
        loss = 0.0
        latents = latents.to(device)
        context = self.pixel_cnn(latents)  # These are the c's (apply pixelCNN to Z's)#[20,2048,6,6]
        targets = self.target_to_32(latents)#32？
        #         targets = latents

        batch_dim, emb_dim, col_dim, row_dim = targets.shape
        targets = targets.reshape(-1, target_dim)

        # Trying to do the arbitrary context vector
        index = np.random.choice(a=[0, 1, 2])
        context = context[:, :, index, :].unsqueeze(3)  # [2, 2048, 6, 1]

        preds_1 = self.conv_1(context).reshape(-1, target_dim) * emb_scale  # key
        preds_2 = self.conv_2(context).reshape(-1, target_dim) * emb_scale
        preds_3 = self.conv_3(context).reshape(-1, target_dim) * emb_scale

        logits_1 = torch.matmul(preds_1, targets.permute(1, 0))  # 12 by 512, 512 by 72 --> 12 by 72
        logits_2 = torch.matmul(preds_2, targets.permute(1, 0))  # value
        logits_3 = torch.matmul(preds_3, targets.permute(1, 0))

        total_elements = batch_dim * row_dim
        b = np.arange(total_elements) / (row_dim)
        b = b.astype(int)
        col = np.arange(total_elements) % (row_dim)

        labels_1 = b * col_dim * row_dim + 3 * row_dim + col  # 直接预测patch的位置吗，太粗暴了
        labels_2 = labels_1 + 6
        labels_3 = labels_2 + 6

        loss += self.loss_func(logits_1, torch.LongTensor(labels_1).to(device))
        loss += self.loss_func(logits_2, torch.LongTensor(labels_2).to(device))
        loss += self.loss_func(logits_3, torch.LongTensor(labels_3).to(device))
        return loss
# Implementation in the paper is unclear.
# I'm going to go with C.

# NCE Loss
# Questions: Is the dimension of Z (B*patches) or (B).
#            I think it's (B, 6, 6, 4096)

class CPCLossNCE(nn.Module):

    def nce_loss(self, z_hat, pos_scores, negative_samples, mask_mat):
        z_hat = z_hat.to(device)
        pos_scores = pos_scores.to(device)
        negative_samples = negative_samples.to(device)
        mask_mat = mask_mat.to(device)

        # (b, 1)
        pos_scores = pos_scores.float()
        batch_size, emb_dim = z_hat.size()
        nb_feat_vectors = negative_samples.size(1) // batch_size  # 36 of them, if 6 by 6 wireframes.

        # (b, b) -> (b, b, nb_feat_vectors)
        # all zeros with ones in diagonal tensor... (ie: b1 b1 are all 1s, b1 b2 are all zeros)
        mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, nb_feat_vectors).float()

        # negative mask
        mask_neg = 1. - mask_pos

        # ----------------------
        # ALL SCORES computation
        # (visualize in your mind a batch size of 2, 36-length segments)
        # (b, dim) x (dim, nb_feats*b) -> (b, b, nb_feats)
        raw_scores = torch.mm(z_hat, negative_samples)
        raw_scores = raw_scores.reshape(batch_size, batch_size, nb_feat_vectors).float()

        # EXTRACT NEGATIVE SCORES
        # (batch_size, batch_size, nb_feat_vectors)
        # HE'S TAKING THE NEGATIVE SAMPLES FROM THE OTHER MINIBATCHES
        # A GIVEN Z_HAT IS ONLY MULTIPLIED BY Z'S FROM OTHER MINIBATCHES
        neg_scores = (mask_neg * raw_scores)
        # ----------------------

        # (b, b, nb_feat_vectors) -> (batch_size, batch_size * nb_feat_vectors)
        neg_scores = neg_scores.reshape(batch_size, -1)
        mask_neg = mask_neg.reshape(batch_size, -1)

        # STABLE SOFTMAX
        # (n_batch_gpu, 1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]

        # DENOMINATOR
        # sum over only negative samples (none from the diagonal)
        neg_sumexp = (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)

        # NUMERATOR
        # compute numerators for the NCE log-softmaxes
        pos_shiftexp = pos_scores - neg_maxes

        # FULL NCE
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()

        return nce_scores

    def forward(self, Z, C, W_list):
        '''
        param Z: latent vecs (B, D, H, W)
        param C: context vecs (B, D, H, W)
        param W_list: list of k-1 W projections
        '''

        # (b, dim, w, h)
        batch_size, emb_dim, h, w = Z.size()

        # (10 x 10 identity matrix)
        diag_mat = torch.eye(batch_size)
        diag_mat = diag_mat.float()

        losses = []
        # calculate loss for each k

        # Below operations preserve raster order (for B, D, H, W) = (1, 5, 2, 2) check.
        # Z_neg holds all z vecs.
        Z_neg = Z.permute(1, 0, 2, 3).reshape(emb_dim, -1)

        for i in range(0, h - 1):
            for j in range(0, w):
                cij = C[:, :, i, j]  # B by D

                for k in range(i + 1, h):  # predict on all vectors in the same column, but below current wireframe.
                    Wk = W_list[str(k)]

                    z_hat_ikj = Wk(cij)
                    zikj = Z[:, :, k, j]

                    # BATCH DOT PRODUCT
                    # (b, d) x (b, d) -> (b, 1)
                    pos_scores = torch.bmm(z_hat_ikj.unsqueeze(1), zikj.unsqueeze(2))
                    pos_scores = pos_scores.squeeze(-1).squeeze(-1)

                    loss = self.nce_loss(z_hat_ikj, pos_scores, Z_neg, diag_mat)
                    losses.append(loss)

        losses = torch.stack(losses)
        loss = losses.mean()
        if np.isnan(loss.item()):
            pdb.set_trace()
            print('boom')
        return loss

def train_raster_patchify(img, size=80, overlap=32):
    '''
    Left-to-right, top to bottom.
    Assumes img is (3, 240, 240).
    '''
    patches = []
    h = -32
    w = -32
    for i in range(6):
        h = h + 32
        for j in range(6):
            w = w + 32
            channel = np.random.randint(3)
            processed_img = np.repeat(np.expand_dims(img[channel, h:h + size, w:w + size], axis=0), 3, axis=0)
            if np.random.randint(2):
                processed_img = np.flip(processed_img, axis=2)
            patches.append(torch.tensor(np.ascontiguousarray(processed_img)))
        w = -32
    return patches

def raster_patchify(img, size=80, overlap=32):
    '''
    Left-to-right, top to bottom.
    Assumes img is (3, 240, 240).
    '''
    patches = []
    h = -32
    w = -32
    for i in range(6):
        h = h + 32
        for j in range(6):
            w = w + 32
            patches.append(img[:, h:h + size, w:w + size])
        w = -32
    return patches

def collate_fn(img_list):
    patches = []
    for img,_ in img_list:
        img_patches = raster_patchify(img)
        patches.append(torch.stack(img_patches))
    return patches#b,36,80,80

def train_collate_fn(img_list):
    patches = []
    for img,_ in img_list:
        img_patches = train_raster_patchify(img)
        patches.append(torch.stack(img_patches))
    return patches

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
def remove_batchnorm(model):
    model.bn1 = Identity()
    model.layer1[0].bn1 = Identity()
    model.layer1[0].bn2 = Identity()
    model.layer1[0].bn3 = Identity()
    model.layer1[0].downsample[1] = Identity()

    model.layer1[1].bn1 = Identity()
    model.layer1[1].bn2 = Identity()
    model.layer1[1].bn3 = Identity()

    model.layer1[2].bn1 = Identity()
    model.layer1[2].bn2 = Identity()
    model.layer1[2].bn3 = Identity()

    model.layer2[0].bn1 = Identity()
    model.layer2[0].bn2 = Identity()
    model.layer2[0].bn3 = Identity()
    model.layer2[0].downsample[1] = Identity()

    model.layer2[1].bn1 = Identity()
    model.layer2[1].bn2 = Identity()
    model.layer2[1].bn3 = Identity()

    model.layer2[2].bn1 = Identity()
    model.layer2[2].bn2 = Identity()
    model.layer2[2].bn3 = Identity()

    model.layer2[3].bn1 = Identity()
    model.layer2[3].bn2 = Identity()
    model.layer2[3].bn3 = Identity()

    model.layer3[0].bn1 = Identity()
    model.layer3[0].bn2 = Identity()
    model.layer3[0].bn3 = Identity()
    model.layer3[0].downsample[1] = Identity()

    model.layer3[1].bn1 = Identity()
    model.layer3[1].bn2 = Identity()
    model.layer3[1].bn3 = Identity()

    model.layer3[2].bn1 = Identity()
    model.layer3[2].bn2 = Identity()
    model.layer3[2].bn3 = Identity()

    model.layer3[3].bn1 = Identity()
    model.layer3[3].bn2 = Identity()
    model.layer3[3].bn3 = Identity()

    model.layer3[4].bn1 = Identity()
    model.layer3[4].bn2 = Identity()
    model.layer3[4].bn3 = Identity()

    model.layer3[5].bn1 = Identity()
    model.layer3[5].bn2 = Identity()
    model.layer3[5].bn3 = Identity()

    #     model.layer4 = Identity()

    model.layer4[0].bn1 = Identity()
    model.layer4[0].bn2 = Identity()
    model.layer4[0].bn3 = Identity()
    model.layer4[0].downsample[1] = Identity()

    model.layer4[1].bn1 = Identity()
    model.layer4[1].bn2 = Identity()
    model.layer4[1].bn3 = Identity()

    model.layer4[2].bn1 = Identity()
    model.layer4[2].bn2 = Identity()
    model.layer4[2].bn3 = Identity()

class CPC(nn.Module):
    def __init__(self):
        super(CPC, self).__init__()
        self.encoder = torchvision.models.resnet50()
        self.encoder.fc = Identity()
        remove_batchnorm(self.encoder)
        self.nce_loss = CPC_loss()

    def forward(self, x, device):
        Z = []
        for img_patches in x:
            img_patches = img_patches.to(device)
            z = self.encoder(img_patches).squeeze()
            z = z.unsqueeze(0).permute(0, 2, 1).reshape(1, 2048, 6, 6)
            Z.append(z)
        Z = torch.stack(Z).squeeze(1)#B,2048,60,60

        loss = self.nce_loss(Z, device)

        return loss
    def save_encoder(self,save_dir,iter_ind,loss):
        torch.save(self.encoder.state_dict(),'./{}/{}_loss{}_encoder_weights.pt'.format(save_dir,iter_ind,loss))
        print('encoder saved ')
class CPC_NCE(nn.Module):
    def __init__(self):
        super(CPC_NCE, self).__init__()
        self.encoder = ResNet_22()
        self.pixel_cnn = PixelCNN(256)
        self.nce_loss = CPC_loss()

        # W transforms (k > 0)
        self.W_list = {}
        for k in range(1, 6):
            w = torch.nn.Linear(256, 256)
            self.W_list[str(k)] = w

        self.W_list = nn.ModuleDict(self.W_list).to(device)

    def forward(self, x):
        Z = []
        C = []
        for img_patches in x:
            img_patches = img_patches.to(device)
            z = self.encoder(img_patches).squeeze()
            z = z.unsqueeze(0).permute(0, 2, 1).reshape(1, 256, 6, 6)
            Z.append(z)
            c = self.pixel_cnn(z)
            C.append(c)

        Z = torch.stack(Z).squeeze(1)
        C = torch.stack(C).squeeze(1)

        loss = self.nce_loss(Z, C, self.W_list)
        return loss
    def save_encoder(self,save_dir,iter_ind):
        torch.save(self.encoder.state_dict(),'./{}/{}_encoder_weights.pt'.format(save_dir,iter_ind))
        print('encoder saved ')

def run_eppoch(iter_num):
    n_imgs=20
    img_num=n_imgs//4
    batch_size=n_imgs
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(240),
        transforms.ColorJitter(brightness=(0.55, 1), contrast=(0.5, 1), saturation=(0.5, 1), hue=0.1),
        transforms.ToTensor(),
    ])

    model = CPC().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 2e-4, weight_decay=1e-5, eps=1e-8)
    imageNet_dataset=our_dataset(data_dir='datafile/ILSVRC2012_img_val', data_csv='data/selected_data.csv', mode='train',
                                 img_num=img_num, transform=data_transform)
    loader = DataLoader(imageNet_dataset, batch_size=batch_size, shuffle=False,collate_fn=train_collate_fn)
    for i, list in enumerate(loader):#list:tuple 2({img list:B},{label list:B})
        if not 0<=i<1:
            continue
        best_loss=10000
        for iter in range(iter_num):
            optimizer.zero_grad()
            loss = model(list,device)

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if iter % 10 == 0:
                if loss < 2.:
                    model.save_encoder('trained_model', i, loss)
                print("Iter: {}/{}, Loss: {}".format(iter, iter_num, loss.item()))

        model.save_encoder('trained_model',i,loss)
if __name__=='__main__':
    torch.cuda.set_device(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_eppoch(500)