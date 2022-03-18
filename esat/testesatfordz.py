import glob
import os
import random
import argparse
import torch.distributed as dist
import numpy as np
import torch
from nystrom_attention import Nystromformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.notebook import tqdm
from model.esat import cv
from lifelines.utils import concordance_index
torch.set_default_tensor_type(torch.FloatTensor)
dist.init_process_group(backend='nccl')
from apex import amp

parser = argparse.ArgumentParser(description='PyTorch TSA Training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--batch_size', default=1, type=int,
                    help='batch size of all GPUs on the current node')
parser.add_argument('--epoch', default=30, type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--seed', default=12, type=int,
                    help='seed for initializing training')
parser.add_argument('--strategy', default=1, type=int,
                    help='different strategy to train')

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
        return img_transformed, float(status), float(surv_time)/365

def CIndex_lifeline(hazards, labels, surv):
    return (concordance_index(surv, hazards[::-1], labels))

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
    torch.cuda.set_device(args.local_rank)
    preparedata(args)

def preparedata(args):
    train_dir="../dongzhong/"
    train_list = glob.glob(os.path.join(train_dir,'*/*.png'))
    train_list, other_list = train_test_split(train_list,
                                              test_size=0.2,
                                              random_state=args.seed)
    valid_list, test_list = train_test_split(other_list,
                                             test_size=0.5,
                                             random_state=args.seed)
    test_transforms = transforms.Compose(
        [
            transforms.Resize((6144, 6144)),
            transforms.ToTensor(),
        ]
    )

    test_data = Datasetprepare(test_list, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
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
        batch_size=args.batch_size
    ).cuda()

    model = amp.initialize(model, opt_level='O2')
    state_dict = torch.load('30e_best_esatdz_bycindex.pth.tar' )
    model.load_state_dict(state_dict)
    model.eval()
    test_pred=[]
    test_status=[]
    test_surv=[]
    for data, status, surv_time in tqdm(test_loader):
        if int(status.numpy()[0])==1:
            data=data.cuda()
            test_status.append(status.numpy()[0])
            test_surv.append(surv_time.numpy()[0])
            pred,_= model(data)
            test_pred.append(pred.cpu().detach().numpy()[0][0])
            print(pred,surv_time.numpy()[0])
    test_cindex= CIndex_lifeline(test_pred, test_status, test_surv)
    print(
        f" cindex : {test_cindex}\n"
    )

if __name__ == '__main__':
    main()