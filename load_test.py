import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps

import torch
import torch.utils.data as data

from bicubic import imresize
from Guassian import Guassian_downsample


def modcrop(img,scale):
    (iw, ih) = img.size
    ih = ih - (ih % scale)
    iw = iw - (iw % scale)
    img = img.crop((0,0,iw,ih))
    return img


class DataloadFromFolderTest(data.Dataset): # load test dataset
    def __init__(self, image_dir, file_list, scale, nFrame, transform):
        super(DataloadFromFolderTest, self).__init__()
        # load image_name from image name list, 
        # note: label list of vimo90k is video name list, not image name list.
        self.image_filenames = glob(os.path.join(image_dir, file_list, 'original/*.png'))
        self.image_filenames.sort()
        self.scale = scale
        self.transform = transform # To_tensor
        self.nFrame = nFrame

    def __getitem__(self, index):

        # load images 
        GT=[]
        for i in range( -self.nFrame//2, self.nFrame//2 + 1):
            ref_index = max(min(index + i, len(self.image_filenames)-1), 1)
            temp = modcrop(Image.open(self.image_filenames[ref_index]).convert('RGB'), self.scale)
            GT.append(np.asarray(temp))
        GT = np.asarray(GT)

        if self.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale), (0,0)), mode='reflect')
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        GT = GT.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']

        GT = GT.view(c,t,h,w)
        target = GT[:,3,:,:]
        LR = Guassian_downsample(GT, self.scale)
        HR = imresize(LR[:,3,:,:], 4,  antialiasing=True) 
        LR_new = []
        group1 = torch.stack((LR[:,0,:,:],LR[:,3,:,:],LR[:,-1,:,:]),1)
        group2 = torch.stack((LR[:,2,:,:],LR[:,3,:,:],LR[:,-3,:,:]),1)
        group3 = torch.stack((LR[:,1,:,:],LR[:,3,:,:],LR[:,-2,:,:]),1)
        LR_new = torch.cat((group1,group2,group3),1) 
        return LR_new, target, HR

        
    def __len__(self):
        # total video number. not image number
        return len(self.image_filenames) 

