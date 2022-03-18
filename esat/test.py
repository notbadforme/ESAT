import numpy as np
import argparse
import torch
from model import surv_model
from torch.utils.data import DataLoader, Dataset
import random
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lifelines.utils import concordance_index
import shutil
import time

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
def save_checkpoint(state, is_best, dst_path='log1/', filename='ckpt.pth.tar'):
    current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    best_name = os.path.join(dst_path, 'best@ep{}_{}_{}.pth.tar'.format(state['epoch'], current_time,state['best_cindex_val']))
    torch.save(state, filename)
    if is_best:
        print('\n\n=> Best val @epoch {}, saving model'.format(state['epoch']))
        shutil.move(filename, best_name)

def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy().reshape(-1)
    hazards = hazards.cpu().numpy().reshape(-1)
    survtime_all = survtime_all.cpu().numpy().reshape(-1)

    label = []
    hazard = []
    surv_time = []

    for i in range(len(hazards)):
        if not np.isnan(hazards[i]) and not np.isnan(survtime_all[i]) and not np.isnan(hazards[i]) :
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])
    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)
    return (concordance_index(new_surv, -new_hazard, new_label))

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='feature')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_dim", type=int, default=256, metavar='MO')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay1', type=float, default=0.001, metavar='M',
                        help='L1 weight_decay')
    parser.add_argument('--weight_decay2', type=float, default=-0.01, metavar='M',
                        help='mim weight_decay')
    parser.add_argument('--epoch', type=int, default=1000, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training')
    args, _ = parser.parse_known_args()
    return args

class Datasetprepare(Dataset):
    def __init__(self,id_list,status_list,time_list,feature_list):
        self.id_list = id_list
        self.status_list = status_list
        self.time_list = time_list
        self.feature_list = feature_list

    def __len__(self):
        self.idlength = len(self.id_list)
        return self.idlength

    def __getitem__(self, idx):
        id = self.id_list[idx]
        feature=self.feature_list[id]
        status= self.status_list[id]
        surv_time=self.time_list[id]
        return feature, status, surv_time

def main(args):
    gpuID = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_cindex_val=-2
    if args["seed"] is not None:
        random.seed(args["seed"])
        os.environ['PYTHONHASHSEED'] = str(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])
        torch.cuda.manual_seed_all(args["seed"])
        torch.backends.cudnn.deterministic = True
    dz_dataset=np.load("NLST_feature.npz")
    feature_list=dz_dataset["features"]
    status_list=dz_dataset["status"]
    time_list=dz_dataset["time"]
    id_list= [x for x in range(len(feature_list))]
    train_list, other_list = train_test_split(id_list,
                                              test_size=0.2,
                                              random_state=args["seed"])
    valid_list, test_list = train_test_split(other_list,
                                             test_size=0.5,
                                             random_state=args["seed"])
    train_data = Datasetprepare(train_list,status_list,time_list,feature_list)
    valid_data = Datasetprepare(valid_list,status_list,time_list,feature_list)
    test_data = Datasetprepare(test_list,status_list,time_list,feature_list)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args["batch_size"])
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args["batch_size"])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args["batch_size"])
    model=surv_model(512,args["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args["lr"],
                                 betas=(0.9, 0.999),
                                 eps=1e-08
                                 )
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=0.5,
                                  patience=10,
                                  mode='min')

    for epoch in range(args["epoch"]):
        model.train()
        pred_risk_all = torch.FloatTensor().to(device)
        surv_time_all = torch.FloatTensor().to(device)
        status_all = torch.FloatTensor().to(device)
        epoch_loss = 0.0
        for step, (feat, surv_time, status) in enumerate(tqdm(train_loader)):
            feat, surv_time, status = feat.to(device), surv_time.to(device), status.to(device)
            optimizer.zero_grad()
            pred_risk = model(feat)
            patient_len = surv_time.shape[0]
            R_matrix = np.zeros([patient_len, patient_len], dtype=int)
            R1_matrix = np.zeros([patient_len, patient_len], dtype=int)
            R2_matrix = np.zeros([patient_len, patient_len], dtype=int)
            for i in range(patient_len):
                for j in range(patient_len):
                    R_matrix[i, j] = surv_time[j] >= surv_time[i]
                    R1_matrix[i, j] = surv_time[j] == surv_time[i]
                    R2_matrix[i, j] = surv_time[j] != surv_time[i]
            R_matrix = torch.FloatTensor(R_matrix).to(device)
            R1_matrix = torch.FloatTensor(R1_matrix).to(device)
            R2_matrix = torch.FloatTensor(R2_matrix).to(device)
            y_status = status.float()
            theta_pred = pred_risk.reshape(-1)
            exp_theta_pred = torch.exp(theta_pred)
            loss_sur = -torch.mean((theta_pred - torch.log(torch.sum(exp_theta_pred * R_matrix, dim=1))) * y_status)
            loss_mim = torch.var(theta_pred * R1_matrix)-torch.var(theta_pred* R2_matrix)
            loss_contra= torch.var(theta_pred* R2_matrix)
            l1_norm = 0.
            for W in model.parameters():
                l1_norm += torch.abs(W).sum()
            loss = loss_sur + args["weight_decay1"] * l1_norm + args["weight_decay2"] * loss_mim
            print(loss_sur, args["weight_decay1"] * l1_norm , args["weight_decay2"] * loss_mim)
            loss.backward()
            optimizer.step()
            surv_time_all = torch.cat([surv_time_all, surv_time])
            status_all = torch.cat([status_all, y_status])
            pred_risk_all = torch.cat([pred_risk_all, pred_risk])
            try:
                c_index = CIndex_lifeline(pred_risk_all.data, status_all, surv_time_all)
            except:
                c_index = -1
            epoch_loss += loss.item()
        print('[train {}/{}] {} -'.format(epoch, args["epoch"], step),
              'Loss: {:.4f} -'.format(epoch_loss / (step + 1)),
              'Cindex: {:.4f} -'.format(c_index))
        #val
        model.eval()
        pred_risk_all = torch.FloatTensor().to(device)
        surv_time_all = torch.FloatTensor().to(device)
        status_all = torch.FloatTensor().to(device)
        epoch_loss = 0.0
        with torch.no_grad():
            for step, (feat, surv_time, status) in enumerate(tqdm(test_loader)):
                feat, surv_time, status = feat.to(device), surv_time.to(device), status.to(device)
                pred_risk = model(feat)
                patient_len = surv_time.shape[0]
                R_matrix = np.zeros([patient_len, patient_len], dtype=int)
                for i in range(patient_len):
                    for j in range(patient_len):
                        R_matrix[i, j] = surv_time[j] >= surv_time[i]
                R_matrix = torch.FloatTensor(R_matrix).to(device)
                y_status = status.float()
                theta_pred = pred_risk.reshape(-1)
                exp_theta_pred = torch.exp(theta_pred)
                loss_sur = -torch.mean((theta_pred - torch.log(torch.sum(exp_theta_pred * R_matrix, dim=1))) * y_status)
                l1_norm = 0.
                for W in model.parameters():
                    l1_norm += torch.abs(W).sum()
                loss = loss_sur + args["weight_decay1"] * l1_norm
                surv_time_all = torch.cat([surv_time_all, surv_time])
                status_all = torch.cat([status_all, y_status])
                pred_risk_all = torch.cat([pred_risk_all, pred_risk])
                try:
                    c_index = CIndex_lifeline(pred_risk_all.data, status_all, surv_time_all)
                except:
                    c_index = -1
                epoch_loss += loss.item()
            print('[val {}/{}] {} -'.format(epoch, args["epoch"], step),
                  'Loss: {:.4f} -'.format(epoch_loss / (step + 1)),
                  'Cindex: {:.4f} -'.format(c_index))
        # scheduler.step(epoch_loss / (step + 1))
        #test
        pred_risk_all = torch.FloatTensor().to(device)
        surv_time_all = torch.FloatTensor().to(device)
        status_all = torch.FloatTensor().to(device)
        with torch.no_grad():
            for step, (feat, surv_time, status) in enumerate(tqdm(test_loader)):
                feat, surv_time, status = feat.to(device), surv_time.to(device), status.to(device)
                pred_risk = model(feat)
                y_status = status.float()
                l1_norm = 0.
                for W in model.parameters():
                    l1_norm += torch.abs(W).sum()
                surv_time_all = torch.cat([surv_time_all, surv_time])
                status_all = torch.cat([status_all, y_status])
                pred_risk_all = torch.cat([pred_risk_all, pred_risk])
        try:
            c_index_test = CIndex_lifeline(pred_risk_all.data, status_all, surv_time_all)
        except:
            c_index_test = -1

        print('[test {}/{}] {} -'.format(epoch, args["epoch"], step),
              'Cindex: {:.4f} -'.format(c_index_test))
        is_best = c_index_test > best_cindex_val
        best_cindex_val = max(c_index_test, best_cindex_val)
        if is_best:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_cindex_val': c_index_test,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                filename='checkpoint@ep{}.pth.tar'.format(epoch))
            print('\n=> Test C-index @ Val_best_C-index: {:.4f}'.format(c_index_test))
    print('\n=> Val_best_C-index: {:.4f}'.format(best_cindex_val))

if __name__ == '__main__':
    args=get_params()
    args = vars(get_params())
    main(args)

