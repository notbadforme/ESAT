import torch
import utils.explainer_utils as util
import torch.nn as nn
import torch.nn.functional as F
from esat import cv
from nystrom_attention import Nystromformer
from torch import optim
from torch.autograd import Variable

class Explainer(object):
    def __init__(self, target_model):
        # ---------------------------
        #   Set Parameter
        # ---------------------------
        self.target_model = target_model

    def globalexplainer(self,input,lr = 0.1, epochs = 20, mask_size=12,weight = 1e-8):
        mask=Variable(torch.ones((mask_size,mask_size)),requires_grad = True)
        optimizer = torch.optim.Adam([mask], lr=lr, weight_decay=0)
        label = self.target_model.Siamesepred(input, input)
        print(label)
        bce_loss = torch.nn.BCELoss()
        for it in range(epochs):
            optimizer.zero_grad()
            gm=util.intialize_mask(input,mask,512)
            prob = self.target_model.Siamesepred(input, gm)
            size_loss = weight*torch.sum(torch.abs(gm))
            loss = bce_loss(prob, torch.Tensor([label]))+size_loss
            loss.backward()
            print(prob,bce_loss(prob, torch.Tensor([label])),size_loss)
            optimizer.step()
        return mask.detach()

    def localexplainer(self,input1,input2,lr = 0.1, epochs = 20, mask_size=96,weight = 1e-8):
        mask1=Variable(torch.ones((mask_size,mask_size)),requires_grad = True)
        mask2=Variable(torch.ones((mask_size,mask_size)),requires_grad = True)
        optimizer = optim.Adam([mask1,mask2], lr=lr, weight_decay=0)
        bce_loss = nn.BCELoss()
        label = self.target_model.Siamesepred(input1, input2)
        print(label)
        for it in range(epochs):
            optimizer.zero_grad()
            lm1=util.intialize_mask(input1,mask1,64)
            lm2=util.intialize_mask(input2,mask2,64)
            prob = self.target_model.Siamesepred(lm1, lm2)
            size_loss = weight * (torch.sum(torch.abs(lm1))+torch.sum(torch.abs(lm2)))
            loss =  bce_loss(prob, torch.Tensor([label]))+size_loss
            print(prob,bce_loss(prob, torch.Tensor([label])),size_loss)
            loss.backward()
            optimizer.step()
        return mask1.detach(),mask2.detach()


class SiameseNet(object):
    def __init__(self):
        # ---------------------------
        #   Set Parameter
        # ---------------------------
        # self.margin = 1.5
        # self.alpha = 0.1
        # self.beta = 2.8
        # self.lr1 , self.lr2, self.lr3= 0.001 , 0.001, 0.01
        efficient_transformer = Nystromformer(
            dim = 512,
            depth = 12,
            heads = 8,
            num_landmarks = 256
        )
        self.model = conViT(
            image_size = 6144,
            patch_size = 16,
            num_classes = 1,
            transformer = efficient_transformer,
            batch_size= 1
        )
        # state_dict = torch.load('0.0001_best_conv_dz_bycindex.pth.tar' ,map_location = 'cpu')
        state_dict = torch.load('20e_e5_conv_nlst_byloss.pth.tar' ,map_location = 'cpu')

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def Siamesepred(self,input1,input2):
        _,embeddings1 = self.model(input1)
        _,embeddings2 = self.model(input2)
        euclidean_distance = F.pairwise_distance(embeddings1.unsqueeze(0), embeddings2.unsqueeze(0),2)
        return torch.sigmoid(euclidean_distance)-0.5





