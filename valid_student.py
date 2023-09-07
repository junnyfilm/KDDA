
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm 
import sys
# from valid_teacher import validation_teacher
# from valid_student import validation_student
import numpy as np
from util import multi_acc 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def student_valid(s_encoder,s_classifier,source_dataloader):
    

    s_encoder=s_encoder.to(device)
    s_classifier=s_classifier.to(device)
        
    criterion = nn.CrossEntropyLoss().to(device)
    vall_acc = 0
    val_loss=[]
    s_encoder.eval()
    s_classifier.eval()

    with torch.no_grad():
        val_acc_num = 0
            
        for X_t,X_t2, X_s,X_s2, y in tqdm(source_dataloader):

            X_s = X_s.float().to(device)
            X_s2 = X_s2.float().to(device)
            y = y.to(device)
            
            raw_f1,fft_f1,s_feature11= s_encoder(X_s,X_s2)
            s_out1, s_logits21, s_a_feature21,_= s_classifier(s_feature11)
            
            
            loss = criterion(s_out1, y)
            val_loss.append(loss.item())      
            val_acc=multi_acc(s_out1,y)
            val_acc_num += val_acc.item()

    return round(sum(val_loss)/len(val_loss),3), val_acc_num/len(source_dataloader)   
        
        
            
    
    
    
    