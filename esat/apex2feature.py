import glob
import os
import argparse
import random
import numpy as np
import torch
from nystrom_attention import Nystromformer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.notebook import tqdm
from model.esat import cv
from lifelines.utils import concordance_index
from apex import amp
os.environ['CUDA_VISIBLE_DEVICE']='0'

parser = argparse.ArgumentParser(description='PyTorch TSA Training')
parser.add_argument('--seed', default=12, type=int,
                    help='seed for initializing training')

class Datasetprepare(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

class Datasetprepare(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img)
        status= img_path.split('/')[-1].split('.')[0].split('_')[-2]
        surv_time=img_path.split('/')[-1].split('.')[0].split('_')[-1]
        return img_transformed, float(status), abs(float(surv_time))

def CIndex_lifeline(hazards, labels, surv):
    return (concordance_index(surv, hazards, labels))

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    preparedata(args)

def preparedata(args):
    train_dir="../dongzhong/"
    all_list = glob.glob(os.path.join(train_dir,'*/*.png'))

    all_transforms = transforms.Compose(
        [
            transforms.Resize((6144, 6144)),
            transforms.ToTensor(),
        ]
    )

    all_data = Datasetprepare([all_list[687]], transform=all_transforms)
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=1)
    efficient_transformer = Nystromformer(
        dim = 128,
        depth = 6,
        heads = 8,
        num_landmarks = 256
    )

    model = cv(
        image_size = 6144,
        patch_size = 16,
        patch_size_big = 384,
        num_classes = 1,
        transformer = efficient_transformer,
        batch_size=1
    ).cuda()

    model = amp.initialize(model, opt_level='O2')
    state_dict = torch.load('30e_best_esatdz_bycindex.pth.tar' )
    model.load_state_dict(state_dict)
    model.eval()
    history_feature = np.load('dz_apex_feature.npz')

    status_list=history_feature["status"].tolist()
    surv_list=history_feature["time"].tolist()
    feature_list=history_feature["features"].tolist()

    # status_list=[]
    # surv_list=[]
    # feature_list=[]
    for data, status, surv_time in tqdm(all_loader):
        data=data.cuda()

        status_list.append(status.numpy()[0])
        surv_list.append(surv_time.numpy()[0])
        pred,feature= model(data)
        feature_list.append(feature.cpu().detach().numpy())
    np.savez("dz_apex_feature", features=feature_list, status=status_list, time=surv_list)
    print(len(status_list))


if __name__ == '__main__':
    main()