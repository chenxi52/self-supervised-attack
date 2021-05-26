import torch
import torch.nn as nn
from selfsupervised import remove_batchnorm,Identity
from resnet import ResNet_22
import torchvision
import numpy as np
import torch.optim as optim
from prep_dataset import our_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

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


def val_raster_patchify(img, size=80, overlap=32):
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
def val_collate_fn(img_list):
    patches = []
    labels = []
    for (img, label) in img_list:
        img_patches = val_raster_patchify(img)
        patches.append(torch.stack(img_patches))
        labels.append(label)

    return patches, labels
def train_collate_fn(img_list):
    patches = []
    labels = []
    for (img, label) in img_list:
        img_patches = train_raster_patchify(img)
        patches.append(torch.stack(img_patches))
        labels.append(label)

    return patches, labels
def one_epoch(patches, model, loss_func, optimizer, device,iter_num,phase='train'):
    x,labels=patches
    labels = torch.from_numpy(np.stack(labels)).to(device)
    for iter in range(iter_num):
        losses = []
        correct = 0
        optimizer.zero_grad()
        preds_logit = model(x, device)
        loss = loss_func(preds_logit, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if iter % 10 == 0:
            print("iter: {}/{},Acc:{}, Loss: {}".format(iter,iter_num , correct/len(x),loss.item()))
        preds_label = torch.argmax(preds_logit, dim=1)
        correct += sum(preds_label == labels)

    return correct, np.mean(losses)


class CPC_Linear(nn.Module):
    def __init__(self):
        super(CPC_Linear, self).__init__()
        self.encoder = torchvision.models.resnet50()
        self.encoder.fc = Identity()
        remove_batchnorm(self.encoder)

        self.bn = nn.BatchNorm2d(2048)
        self.conv_1 = nn.Conv2d(2048, 512, (1, 1))
        self.avg_pool = nn.AvgPool2d(6, 6)

        self.bn_50_1 = nn.BatchNorm1d(50)
        self.bn_50_2 = nn.BatchNorm1d(50)

        self.lin_1 = nn.Linear(512 * 6 * 6, 50)
        self.relu = nn.ReLU()
        self.lin_2 = nn.Linear(50, 50)
        self.lin_3 = nn.Linear(50, 10)

    #         self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, device):
        Z = []
        for img_patches in x:
            img_patches = img_patches.to(device)#36,3,80,80
            z = self.encoder(img_patches).squeeze()#36,2048
            z = z.unsqueeze(0).permute(0, 2, 1).reshape(1, 2048, 6, 6)
            Z.append(z)

        Z = torch.stack(Z).squeeze(1)#20,2048,6,6

        x = self.conv_1(self.bn(Z))
        x = x.view(-1, 512 * 6 * 6)

        x = self.relu(self.bn_50_1(self.lin_1(x)))
        x = self.relu(self.bn_50_2(self.lin_2(x)))

        output = self.lin_3(x)
        # output = self.avg_pool(self.conv_1(self.bn(Z))).squeeze(2).squeeze(2)

        return output
def run_epochs(iter_num):
    torch.cuda.set_device(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CPC_Linear()
    pretrained_dict = torch.load('./trained_model/0_loss0.20758995413780212_encoder_weights.pt')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

    # ----------------------------------------------
    # FREEZE ENCODER
    for param in model.encoder.parameters():
        param.requires_grad = False

    n_imgs = 20
    img_num = n_imgs // 4
    batch_size = n_imgs
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
    ])
    imageNet_dataset = our_dataset(data_dir='datafile/ILSVRC2012_img_val', data_csv='data/selected_data.csv', mode='train',
                                   img_num=img_num, transform=data_transform)
    train_dl = DataLoader(imageNet_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collate_fn)
    val_dl=DataLoader(imageNet_dataset,batch_size=batch_size,shuffle=True,collate_fn=val_collate_fn)
    for i,list in enumerate(train_dl):
        if not 0<=i<1:
            continue

        correct,epoch_loss = one_epoch(list, model, loss_func, optimizer, device,iter_num, phase='train')
    # print("Average Epoch {} Loss: {}".format(i, epoch_loss))
    # correct, _ = one_epoch(train_dl, model, loss_func, optimizer, device, phase='val')
    # print("Train Accuracy: {}".format(1. * correct ))
    # correct, _ = one_epoch(val_dl, model, loss_func, optimizer, device, phase='val')
    # print("Validation Accuracy: {}".format(1. * correct ))

    if i in [1, 10, 20, 30]:
        torch.save(model.state_dict(), "pretrained_frozen_acc_{}_{}.pt".format(correct,i))

if __name__=='__main__':
    run_epochs(100)