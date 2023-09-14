import random
import torch
import math
import numpy as np
import cv2

def cut_patch(mb_img):#for numpy
    h, w,c = mb_img.shape
    top = random.randrange(0, round(h))
    bottom = top + random.randrange(round(h * 0.8), round(h * 0.9))
    left = random.randrange(0, round(w))
    right = left + random.randrange(round(h * 0.8), round(h * 0.9))
    if (bottom - top) % 2 == 1:
        bottom -= 1
    if (right - left) % 2 == 1:
        right -= 1
    return mb_img[top:bottom, left:right,:]

def paste_patch(img, patch):
    imgh,imgw,imgc = img.shape
    patchh, patchw,patchc = patch.shape
    angle = random.randrange(-2 * round(math.pi), 2 * round(math.pi))
    #scale = random.randrange(0, 1)
    scale=1
    affinematrix=np.float32([[scale * math.cos(angle),scale * -math.sin(angle),0],[scale * math.sin(angle), scale * math.cos(angle), 0]])
    affinepatch=cv2.warpAffine(patch,affinematrix,(patchw,patchh))
    patch_h_position = random.randrange(1, round(imgh) - round(patchh) - 1)
    patch_w_position = random.randrange(1, round(imgw) - round(patchw) - 1)
    pasteimg = np.copy(img)
    pasteimg[patch_h_position:patch_h_position + patchh,
    patch_w_position:patch_w_position + patchw,:] = affinepatch
    mask=np.zeros((img.shape[0],img.shape[1],1))
    mask[patch_h_position:patch_h_position + patchh,
    patch_w_position:patch_w_position + patchw] = 1
    return pasteimg,mask