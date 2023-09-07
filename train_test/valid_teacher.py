import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm 
from utils.util import multi_acc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validation_teacher(encoder,temporal_c,temporal_v,spatial_c,spatial_v,cross_att,classifier, val_loader, criterion, device):
    encoder.eval()
    temporal_c.eval()
    temporal_v.eval()
    spatial_c.eval()
    spatial_v.eval()
    cross_att.eval()
    classifier.eval()

    val_loss = []
    pred_labels = []
    true_labels = []
    threshold = 0.35
    vlacc=[]
    
    with torch.no_grad():
        val_acc_num = 0


            
        for X_t,X_t2, X_s,X_s2, y in tqdm(val_loader):
            X_t = X_t.float().to(device)
            X_t2 = X_t2.float().to(device)
            X_s = X_s.float().to(device)
            X_s2 = X_s2.float().to(device)
            y = y.to(device)
            
            t_v_r,t_c_r,t_v_f,t_c_f= encoder(X_t,X_t2,X_s,X_s2)
            t_c_r=temporal_c(t_c_r)
            t_v_r=temporal_v(t_v_r)
            t_c_f=spatial_c(t_c_f)
            t_v_f=spatial_v(t_v_f)
            att_r,att_f=torch.cat([t_c_r,t_c_f], dim=1),torch.cat([t_v_r,t_v_f], dim=1)
            att=cross_att(att_r,att_f)
            out, logits2,feature= classifier(att)
            
            
            
            loss = criterion(out, y)
            val_loss.append(loss.item())      
            val_acc=multi_acc(out,y)
            val_acc_num += val_acc.item()


            
    return val_loss, val_acc_num/len(val_loader)   