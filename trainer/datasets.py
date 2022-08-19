import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import cv2
import SimpleITK as sitk
import torch.nn.functional as F

def load_norm(filepath, nchannels=3):
    if nchannels == 1:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    elif nchannels == 7 and filepath.split('.')[-1] in ('jpg', 'png'):
        og = cv2.imread(filepath)
        img = np.zeros((nchannels, og.shape[0], og.shape[1]), dtype=np.float32)
        for i in range(nchannels):
            img[i,:,:] = og[:,:,i%3]
        # sitk.WriteImage(sitk.GetImageFromArray(img), './tmp.tif')
        # img = sitk.GetArrayFromImage(sitk.ReadImage('./tmp.tif'))

    elif nchannels == 7:
        img = sitk.ReadImage(filepath)
        img = sitk.GetArrayFromImage(img)
        if len(img.shape) == 2:
            size = min(img.shape)
            length = max(img.shape)
            assert length % size == 0 and length // size == 8
            resized = np.zeros((nchannels, size, size), dtype=np.float32)
            for i in range(nchannels):
                sl = img[i*size:(i+1)*size, :]
                thresh = np.percentile(sl, 99)
                sl[sl > thresh] = thresh
                resized[i,:,:] = sl
            img = resized
    else:
        img = cv2.imread(filepath)
    
    if img.shape[0] != nchannels:
        resized = np.zeros((nchannels, img.shape[0], img.shape[1]), dtype=np.float32)
        for i in range(nchannels):
            resized[i,:,:] = img[:,:,i]
        img = resized
    img_norm = cv2.normalize(img, None, -1, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return torch.from_numpy(img_norm.astype(np.float32))

def get_datasets(sourcepath, targetpath, input_nc, output_nc, size):
    lstA = sorted(glob.glob(f'{sourcepath}/*'))
    lstB = sorted(glob.glob(f'{targetpath}/*'))
    fnamesA = [x.split('/')[-4][-1] + os.path.basename(x).split('.')[0] for x in lstA]
    fnamesB = [x.split('/')[-4][-1] + os.path.basename(x).split('.')[0] for x in lstB]
    intersection = set(fnamesA) & set(fnamesB)
    files_A = [x for x in lstA if x.split('/')[-4][-1] + os.path.basename(x).split('.')[0] in intersection]
    files_B = [x for x in lstB if x.split('/')[-4][-1] + os.path.basename(x).split('.')[0] in intersection]
    # files_A = lstA
    # files_B = lstB
    assert len(files_A) == len(files_B)
    
    cut = int(len(files_A) * 0.9)
    train_data = ImageDataset(files_A[:cut], files_B[:cut], input_nc, output_nc, size)
    val_data = ImageDataset(files_A[cut:], files_B[cut:], input_nc, output_nc, size)

    # train_data = ImageDataset(files_A[:3000], files_B[:3000], input_nc, output_nc, size)
    # val_data = ImageDataset(files_A[3000:3500], files_B[3000:3500], input_nc, output_nc, size)

    return train_data, val_data


class ImageDataset(Dataset):
    def __init__(self, fA, fB, input_nc, output_nc, size, 
                unaligned=False):
        self.files_A = fA
        self.files_B = fB
        assert len(self.files_A) == len(self.files_B)

        self.unaligned = unaligned
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.size = size

    def __getitem__(self, index):
        item_A = load_norm(self.files_A[index % len(self.files_A)], self.input_nc)
        item_B = load_norm(self.files_B[index % len(self.files_B)], self.output_nc)
        item_A = F.interpolate(item_A.unsqueeze(0), size=(self.size, self.size)).squeeze(0)
        item_B = F.interpolate(item_B.unsqueeze(0), size=(self.size, self.size)).squeeze(0)
        # print(item_A.shape, item_B.shape) # torch.Size([3, 7664, 7664]) torch.Size([7, 2540, 2539])
        
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
