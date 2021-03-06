import csv
import numpy as np
import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset,DataLoader

class our_dataset(Dataset):
    def __init__(self,data_dir, data_csv,mode,img_num,transform):
        assert mode in ['train','attack','all'],'WRONG DATASET MODE'
        #assert img_num in [1,5,10,20],'ONLT SUPPORT 2/10/20/40 IMAGES'
        super(our_dataset, self).__init__()
        self.mode=mode
        self.data_dir=data_dir
        data_csv=open(data_csv,'r')
        cdvreader=csv.reader(data_csv)
        data_ls=list(cdvreader)
        self.imgs=self.prep_imgs_dir(data_ls,img_num)
        self.transform=transform
    def prep_imgs_dir(self,data_ls,nImg):
        imgs_ls=[]
        if self.mode in ['train','attack']:
            if nImg>=10:
                sel_ls = list(range(nImg))
                imgs_ls += self.mk_img_ls(data_ls, sel_ls)
            elif nImg == 1:
                for jkl in list(range(10)):
                    imgs_ls += self.mk_img_ls(data_ls, [jkl])
            elif nImg == 5:#这里没懂
                sel_ls_1 = list(range(5))
                sel_ls_2 = list(range(5, 10))
                imgs_ls += self.mk_img_ls(data_ls, sel_ls_1)
                imgs_ls += self.mk_img_ls(data_ls, sel_ls_2)
        elif self.mode == 'all':
            sel_ls = list(range(50))
            imgs_ls += self.mk_img_ls(data_ls, sel_ls)
        return imgs_ls

    def mk_img_ls(self, data_ls, sel_ls):
        #每类选nImags个
        #imgs_ls[[img_dir,label_ind],[],[]
        imgs_ls = []
        for label_ind in range(len(data_ls)):#csv的行数，类的数量
            for img_ind in sel_ls:#nImages
                imgs_ls.append(
                    [self.data_dir + '/' + data_ls[label_ind][0] + '/' + data_ls[label_ind][1 + img_ind],
                     label_ind])
        return imgs_ls

    def __getitem__(self, item):
        img = Image.open(self.imgs[item][0])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img), self.imgs[item][1]

    def __len__(self):
        return len(self.imgs)

class imagenet_ds(Dataset):
    def __init__(self, data_dir, data_csv, mode, img_num, transform):
        assert mode in ['train', 'attack', 'all'], 'WRONG DATASET MODE'
        #assert img_num in [1, 5, 10, 20], 'ONLT SUPPORT 2/10/20/40 IMAGES'
        super(our_dataset, self).__init__()
        self.mode = mode
        self.data_dir = data_dir
        data_csv = open(data_csv, 'r')
        cdvreader = csv.reader(data_csv)
        data_ls = list(cdvreader)
        self.imgs = self.prep_imgs_dir(data_ls, img_num)
        self.transform = transform

    def prep_imgs_dir(self, data_ls, nImg):
        imgs_ls = []
        if self.mode in ['train', 'attack']:
            if nImg >= 10:
                sel_ls = list(range(nImg))
                imgs_ls += self.mk_img_ls(data_ls, sel_ls)
            elif nImg == 1:
                for jkl in list(range(10)):
                    imgs_ls += self.mk_img_ls(data_ls, [jkl])
            elif nImg == 5:  # 这里没懂
                sel_ls_1 = list(range(5))
                sel_ls_2 = list(range(5, 10))
                imgs_ls += self.mk_img_ls(data_ls, sel_ls_1)
                imgs_ls += self.mk_img_ls(data_ls, sel_ls_2)
        elif self.mode == 'all':
            sel_ls = list(range(50))
            imgs_ls += self.mk_img_ls(data_ls, sel_ls)
        return imgs_ls

    def mk_img_ls(self, data_ls, sel_ls):
        # 每类选nImags个
        # imgs_ls[[img_dir,label_ind],[],[]
        imgs_ls = []
        for label_ind in range(len(data_ls)):  # csv的行数，类的数量
            for img_ind in sel_ls:  # nImages
                imgs_ls.append(
                    [self.data_dir + '/' + data_ls[label_ind][0] + '/' + data_ls[label_ind][1 + img_ind],
                     label_ind])
        return imgs_ls

    def __getitem__(self, item):
        img = Image.open(self.imgs[item][0])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img), self.imgs[item][1]

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    # One way to prepare 'data/selected_data.csv'
    # selected_data_csv = open('data/selected_data.csv', 'w')
    # data_writer = csv.writer(selected_data_csv)
    dataset_dir = 'data/ILSVRC2012_img_val'
    dataset = torchvision.datasets.ImageFolder(dataset_dir)
    label_ind = torch.randperm(500).numpy()
    selected_labels_ls = np.array(dataset.classes)[label_ind]
    # for label_name in selected_labels_ls:
    #     data_writer.writerow([label_name]+os.listdir(os.path.join(dataset_dir, label_name)))