import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm 
import sys
from train_test.valid_teacher import validation_teacher
import numpy as np


T1 = 1200
T2 = 4800
af = 0.5
step_per_epoch = 60
total_steps = 6000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def alpha_weight(step):
    if step < T1:
        return 0.9
    else:
        return max(af, 0.7 ** ((step-T1) / (step_per_epoch*(total_steps-T1)/(T2-T1))))



def teacher_train(encoder,temporal_c,temporal_v,spatial_c,spatial_v,cross_att,classifier, optimizer, train_loader1,train_loader2, val_loader1,val_loader2, scheduler, device,epochs):
    step=0
    
    encoder=encoder.to(device)
    temporal_c=temporal_c.to(device)
    temporal_v=temporal_v.to(device)
    spatial_c=spatial_c.to(device)
    spatial_v=spatial_v.to(device)
    cross_att=cross_att.to(device)
    classifier=classifier.to(device)

    best_score = 0
    best_encoder = None
    best_temp = None
    best_spec = None
    best_crossatt = None
    best_classifier = None
    criterion = nn.CrossEntropyLoss().to(device)
    kldiv = nn.KLDivLoss().to(device)

    for epoch in range(epochs):
        train_loss = []
        train_loss2 = []
  
        encoder.train()
        temporal_c.train()
        temporal_v.train()
        spatial_c.train()
        spatial_v.train()
        cross_att.train()
        classifier.train()
        print("epoch: ",epoch)
        print("source 1 train")
        for X_t,X_t2, X_s,X_s2, y in tqdm(train_loader1):
            X_t = X_t.float().to(device)
            X_t2 = X_t2.float().to(device)
            X_s = X_s.float().to(device)
            X_s2 = X_s2.float().to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            t_v_r,t_c_r,t_v_f,t_c_f= encoder(X_t,X_t2,X_s,X_s2)
            t_c_r=temporal_c(t_c_r)
            t_v_r=temporal_v(t_v_r)
            t_c_f=spatial_c(t_c_f)
            t_v_f=spatial_v(t_v_f)
            att_c,att_v=torch.cat([t_c_r,t_c_f], dim=1),torch.cat([t_v_r,t_v_f], dim=1)
            att=cross_att(att_c,att_v)
            out, logits2,feature= classifier(att)
            
            time = kldiv(nn.functional.log_softmax(t_c_r, dim=1),nn.functional.softmax( t_v_r, dim=1))
            fft = kldiv(nn.functional.log_softmax(t_c_f, dim=1),nn.functional.softmax( t_v_f, dim=1))
            self_distill=time+fft
            class_loss = criterion(out, y)
            
            loss = alpha_weight(step)*self_distill+(1-alpha_weight(step))*class_loss

            loss.backward()
            
            optimizer.step()
            step += 1

            train_loss.append(loss.item())
        val_loss, val_score = validation_teacher(encoder,temporal_c,temporal_v,spatial_c,spatial_v,cross_att,classifier, val_loader1, criterion, device)
        print(f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss) :.5f}] Val Loss : [{np.mean(val_loss) :.5f}] Val ACC Score : [{val_score:.5f}]')
        
        print("source 2 train")
        encoder.train()
        temporal_c.train()
        temporal_v.train()
        spatial_c.train()
        spatial_v.train()
        cross_att.train()
        classifier.train()
        for X_t,X_t2, X_s,X_s2, y in tqdm(train_loader2):
            X_t = X_t.float().to(device)
            X_t2 = X_t2.float().to(device)
            X_s = X_s.float().to(device)
            X_s2 = X_s2.float().to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            t_v_r,t_c_r,t_v_f,t_c_f= encoder(X_t,X_t2,X_s,X_s2)
            t_c_r=temporal_c(t_c_r)
            t_v_r=temporal_v(t_v_r)
            t_c_f=spatial_c(t_c_f)
            t_v_f=spatial_v(t_v_f)
            att_c,att_v=torch.cat([t_c_r,t_c_f], dim=1),torch.cat([t_v_r,t_v_f], dim=1)
            att=cross_att(att_c,att_v)
            out, logits2,feature= classifier(att)
            
            time = kldiv(nn.functional.log_softmax(t_c_r, dim=1),nn.functional.softmax( t_v_r, dim=1))
            fft = kldiv(nn.functional.log_softmax(t_c_f, dim=1),nn.functional.softmax( t_v_f, dim=1))
            self_distill=time+fft
            class_loss = criterion(out, y)
            
            loss = alpha_weight(step)*self_distill+(1-alpha_weight(step))*class_loss

            loss.backward()
            
            optimizer.step()
            step += 1

            train_loss2.append(loss.item())

        val_loss2, val_score2 = validation_teacher(encoder,temporal_c,temporal_v,spatial_c,spatial_v,cross_att,classifier, val_loader2, criterion, device)
        print(f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss2) :.5f}] Val Loss : [{np.mean(val_loss2) :.5f}] Val ACC Score : [{val_score2:.5f}]')
        val_score3=val_score2+val_score
        if scheduler is not None:
            scheduler.step(val_score3)
            
        if best_score <= val_score3:
            
            best_encoder = encoder
            best_temp_c = temporal_c
            best_temp_v = temporal_v
            best_spec_c = spatial_c
            best_spec_v = spatial_v
            best_crossatt= cross_att
            best_classifier = classifier
            
            best_score = val_score3
        

    return best_encoder,best_temp_c,best_temp_v,best_spec_c,best_spec_v,best_crossatt,best_classifier
        
    