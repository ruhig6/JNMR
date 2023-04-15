import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class DBreader_Vimeo90k(Dataset):
    def __init__(self, root, path_list, random_crop=None, resize=None, augment_s=None, augment_t=None):
        self.root = root
        self.path_list = path_list

        self.random_crop = random_crop
        self.resize = resize
        self.augment_s = augment_s
        self.augment_t = augment_t

    def __getitem__(self, index):
        path = self.path_list[index]
        return self.Vimeo90K_loader(path)

    def __len__(self):
        return len(self.path_list)

    def Vimeo90K_loader(self, im_path):
        abs_im_path = os.path.join(self.root, 'sequences', im_path)

        transform_list = []
        if self.resize is not None:
            transform_list += [transforms.Resize(self.resize)]
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        
        rawFrame0 = Image.open(os.path.join(abs_im_path, "im1.png"))
        rawFrame1 = Image.open(os.path.join(abs_im_path, "im3.png"))
        rawFrame2 = Image.open(os.path.join(abs_im_path, "im4.png"))
        rawFrame3 = Image.open(os.path.join(abs_im_path, "im5.png"))
        rawFrame4 = Image.open(os.path.join(abs_im_path, "im7.png"))

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame2, output_size=self.random_crop)

            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)
            rawFrame3 = TF.crop(rawFrame3, i, j, h, w)
            rawFrame4 = TF.crop(rawFrame4, i, j, h, w)

        if self.augment_s:
            if random.randint(0, 1):

                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
                rawFrame3 = TF.hflip(rawFrame3)
                rawFrame4 = TF.hflip(rawFrame4)

            if random.randint(0, 1):

                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)
                rawFrame3 = TF.vflip(rawFrame3)
                rawFrame4 = TF.vflip(rawFrame4)
               

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)
        frame3 = self.transform(rawFrame3)
        frame4 = self.transform(rawFrame4)

        if self.augment_t:
            if random.randint(0, 1):
                return frame4, frame3, frame2, frame1, frame0
            else:
                return frame4, frame3, frame2, frame1, frame0
        else:
            return frame0, frame1, frame2, frame3, frame4


def make_dataset(root, list_file):
    raw_im_list = open(os.path.join(root, list_file)).read().splitlines()
    raw_im_list = raw_im_list[:]  # the last line is invalid in test set
    assert len(raw_im_list) > 0
    random.shuffle(raw_im_list)
    return raw_im_list


def Vimeo90K_interp(root):
    train_list = make_dataset(root, "sep_trainlist.txt")
    val_list = make_dataset(root, "sep_testlist.txt")
    train_dataset = DBreader_Vimeo90k(root, train_list)
    val_dataset = DBreader_Vimeo90k(root, val_list)
    return train_dataset, val_dataset

def Vimeo90K_interp_test(root):
    val_list = make_dataset(root, "sep_vallist.txt")
    test_list = make_dataset(root, "sep_testlist.txt")
    val_dataset = DBreader_Vimeo90k(root, val_list)
    test_dataset = DBreader_Vimeo90k(root, test_list)
    return val_dataset, test_dataset