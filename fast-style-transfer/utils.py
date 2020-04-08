import cv2
import numpy as np
import matplotlib.pylost as plt
import torch
from torchvision import transforms, datasets

# Gram Matrix
def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H * W)
    x_t = x.transpose(1, 2)
    # torch.bmm 计算两个tensor矩阵的乘法，维度必须为3
    return torch.bmm(x, x_t) / (C * H * W)

def load_image(path):
    # load as BGR
    img = cv2.imread(path)
    return img

def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imshow() only accepts float [0,1] or int[0,255]
    img = np.array(img/255).clip(0, 1)
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()

def saveimg(img, image_path):
    img = img.clip(0, 255)
    cv2.imwrite(image_path, img)

def img2tensor(img, max_size=None):
    if (max_size==None):
        img2ten_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    else:
        H, W, C = img.shape
        img_size = tuple([int((float(max_size) / max([H, W]))*x) for x in [H, W]])
        img2ten_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    tensor = img2ten_t(img)
# unsqueeze: 去掉维数为1的维度
    tensor = tensor.unsqueeze(dim=0)
    return tensor

def tensor2img(tensor):
    # 去掉batch_size的维度
    tensor = tensor.squeeze()
    img = tensor.numpy()
    # [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img

# reference : https://github.com/rrmina/fast-neural-style-pytorch/blob/master/utils.py
def transfer_color(src, dest):
    # Transfer Color using YIQ colorspace. Useful in preserving colors in style transfer.
    src, dest = src.clip(0, 255), dest.clip(0, 255)

    # Resize src to dest's size
    H, W, _ = src.shape
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_CUBIC)

    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)  # 1 Extract the Destination's luminance
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)  # 2 Convert the Source from BGR to YIQ/YCbCr
    src_yiq[..., 0] = dest_gray  # 3 Combine Destination's luminance and Source's IQ/CbCr

    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0, 255)  # 4 Convert new image from YIQ back to BGR

# 画 loss 的图像...


