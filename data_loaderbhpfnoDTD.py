import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
from CutpasteAugment import cut_patch,paste_patch

class MVTecTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        #self.images = sorted(glob.glob(root_dir+"/*/*.JPG")) # Please activate this when using the VisA dataset.
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
        imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #fft
        f = np.fft.fft2(imagegray)
        fshift = np.fft.fftshift(f)

        #BHPF
        rows, cols = imagegray.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        d = 30 #cutoff frequency
        n = 2 #BHPF order
        epsilon = 1e-6 #avoid dividing zero
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
        maska = 1 / (1 + (d / (dist + epsilon)) ** (2 * n))
        fshift_filtered = fshift * maska

        #inverse fft
        f_ishift = np.fft.ifftshift(fshift_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        imagegray = np.real(image_filtered).astype(np.float32)

        imagegray=imagegray[:,:,None]
        image = np.transpose(image, (2, 0, 1))
        imagegray = np.transpose(imagegray, (2, 0, 1))

        mask = np.transpose(mask, (2, 0, 1))
        return image, mask,imagegray

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask,imagegray = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            #mask_file_name = file_name.split(".")[0]+".png" Please activate this when using the VisA dataset.
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask,imagegray = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx,'imagegray':imagegray}

        return sample



class MVTecTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        self.anomaly_source_paths = self.image_paths

        #self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                           iaa.WithColorspace(to_colorspace="HSV",from_colorspace="RGB",children=iaa.WithChannels(0,iaa.Add((0,90)))),
                           iaa.Dropout2d(p=0.5),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def augment_cutpaste(self, image):
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((image.shape[0],image.shape[1],1),dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            patch = cut_patch(image)
            augmented_image,msk=paste_patch(image,patch)
            msk=msk.astype(np.float32)
            augmented_image=augmented_image.astype(np.float32)
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        what_anomaly = torch.rand(1).numpy()[0]
        if what_anomaly > 0.5:
            augmented_image, anomaly_mask, has_anomaly = self.augment_cutpaste(image)
        else:
            augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        auggray = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)

        #fft
        f = np.fft.fft2(auggray)
        fshift = np.fft.fftshift(f)

        #BHPF
        rows, cols = auggray.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        d = 30  #cutoff frequency
        n = 2 #BHPF order
        epsilon = 1e-6 #avoid dividing zero
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
        mask = 1 / (1 + (d / (dist + epsilon)) ** (2 * n))
        fshift_filtered = fshift * mask

        #inverse fft
        f_ishift = np.fft.ifftshift(fshift_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        auggray = np.real(image_filtered).astype(np.float32)

        auggray=auggray[:,:,None]
        image = np.transpose(image, (2, 0, 1))
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        auggray = np.transpose(auggray, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly,auggray

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly,auggray = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx,'auggray':auggray}

        return sample
