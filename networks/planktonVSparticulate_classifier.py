import os
import shutil
import glob
import PIL.Image as Image
import argparse
import torch
import torchvision.transforms as transforms
import DLmodel
from tqdm import tqdm
import sys
from torch.utils.data import Dataset
import torch.nn.functional as F


class raw_data(Dataset):
    def __init__(self, folder_name, transform=None):
        imgs = glob.glob(os.path.join(folder_name, '*.jpg'))
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])

        # (height, width, channels) = img.shape
        # if channels != 3:  # 因为图像文件的原因，可能一些图像是4通道的，所以强行提取3通道
        #     img = img[:, :, 0:3]

        if self.transform is not None:
            img = self.transform(img)

        imgfile = self.imgs[idx]  # 图像的文件名
        return img, imgfile


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This is a two-class classifier which can classify the plankton and non-plankton images, and then move them to right directories automatically.')
    parser.add_argument('--folder', type=str, required=True, help='The images folder you want to process.')
    parser.add_argument('--dst', default='/home/nvidia/PlanktonDatabase/', type=str, required=False,
                        help='The destination directory you want the plankton images to go.')
    parser.add_argument('--prob_thresh', default=0.8, type=float, required=False,
                        help='The confidence probability threshold of judging a image as a plankton.')

    args = parser.parse_args()
    folder = args.folder
    PROB_THRESH = args.prob_thresh

    rawdata = raw_data(
        folder_name=folder,
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize([256, 256]),
            transforms.CenterCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.0257, 0.0257, 0.0257), std=(0.0869, 0.0869, 0.0869))
        ])
    )
    raw_loader = torch.utils.data.DataLoader(
        dataset=rawdata,
        batch_size=32,
        shuffle=False,
        num_workers=1
    )

    model = DLmodel.model.to(DEVICE)
    model.load_state_dict(torch.load('vgg11_bn_10.pt'))
    model.eval()

    plankton_folder = os.path.join(args.dst, folder)

    with torch.no_grad():
        for batch_idx, (imgdata, imgfile) in enumerate(tqdm(raw_loader)):
            imgdata = imgdata.to(DEVICE)
            out = model(imgdata)
            top2 = F.softmax(out,dim=1).topk(2)[1]
            prob = F.softmax(out,dim=1).max(1)[0]
            for idx in range(len(imgdata)):
                res_class = int(top2[idx][0])
                res_prob = float(prob[idx])

                dst_folder = ''
                if res_class == 2:
                    if res_prob > PROB_THRESH:
                        dst_folder = plankton_folder  # 将判别为浮游生物的图像移动到指定文件夹下
                    else:
                        res_class = res_class = int(top2[idx][1])
                        dst_folder = os.path.join(folder, str(res_class))
                else:
                    dst_folder = os.path.join(folder, str(res_class))

                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)

                filename = os.path.split(imgfile[idx])[-1]
                shutil.move(src=imgfile[idx], dst=os.path.join(dst_folder, filename))
