import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets
from fft import FFT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset_distill(Dataset):
    def __init__(self, vib_x,cur_x, data_y):
        super(CustomDataset_distill, self).__init__()
        self.data_X1 = vib_x
        self.data_X2 = cur_x
        self.data_y = data_y
        
    def __len__(self):
        return len(self.data_X1)
    
    def __getitem__(self, index):
        # 지식 증류 학습 시
        teacher_X = torch.Tensor(self.data_X1[index])
        student_X = torch.Tensor(self.data_X2[index])
        
        
        teacher_fft = FFT(teacher_X[0],6760)
        teacher_fft = torch.from_numpy(teacher_fft)
        teacher_fft= torch.unsqueeze(teacher_fft, 0)
        
        student_fft = FFT(student_X[0],50100)
        student_fft = torch.from_numpy(student_fft)
        student_fft= torch.unsqueeze(student_fft, 0)
        
        
        y = self.data_y[index]
        return teacher_X.to(device).float(),teacher_fft.to(device).float(), student_X.to(device).float(),student_fft.to(device).float(), y.to(device)

# class CustomDataset_distill_triplet(Dataset):
#     def __init__(self, vib_x, cur_x, data_y, train_model="teacher"):
#         self.data_X1 = vib_x
#         self.data_X2 = cur_x
#         self.data_y = data_y
#         self.index = torch.arange(len(data_y))
#         self.train_model = train_model
#     def __len__(self):
#         return len(self.data_X1)
    
#     def __getitem__(self, item):
#         anchor_label = self.data_y[item]
        
#         # Anchor samples
#         anchor_vib = torch.Tensor(self.data_X1[item])
#         anchor_cur = torch.Tensor(self.data_X2[item])

#         # Find the list of positive and negative samples
#         positive_list = self.index[self.index != item][self.data_y[self.index != item] == anchor_label]
#         negative_list = self.index[self.index != item][self.data_y[self.index != item] != anchor_label]
        
#         # Choose one positive and one negative sample randomly
#         positive_item = random.choice(positive_list)
#         negative_item = random.choice(negative_list)
        
#         # Positive samples
#         # positive_vib = torch.Tensor(self.data_X1[positive_item])
#         positive_cur = torch.Tensor(self.data_X2[positive_item])
        
#         # Negative samples
#         # negative_vib = torch.Tensor(self.data_X1[negative_item])
#         negative_cur = torch.Tensor(self.data_X2[negative_item])

#         # Add FFT
#         anchor_vib_fft = torch.unsqueeze(torch.from_numpy(FFT(anchor_vib[0], 6760)),0)
#         anchor_cur_fft = torch.unsqueeze(torch.from_numpy(FFT(anchor_cur[0], 50100)),0)

#         positive_cur_fft = torch.unsqueeze(torch.from_numpy(FFT(positive_cur[0], 50100)),0)
#         negative_cur_fft = torch.unsqueeze(torch.from_numpy(FFT(negative_cur[0], 50100)),0)


#         train_model=self.train_model
#         if train_model=="teacher":
#             return anchor_vib.to(device).float(), anchor_vib_fft.to(device).float(), anchor_cur.to(device).float(), anchor_cur_fft.to(device).float(),anchor_label.to(device)
#         else:
#             return anchor_vib.to(device).float(), anchor_vib_fft.to(device).float(), anchor_cur.to(device).float(), anchor_cur_fft.to(device).float(),anchor_label.to(device), positive_cur.to(device).float(),positive_cur_fft.to(device).float(),negative_cur.to(device).float(),negative_cur_fft.to(device).float()

#         # return {
#         #     'anchor_vib': anchor_vib, 'anchor_cur': anchor_cur, 'anchor_label': anchor_label,
#         #     'positive_vib': positive_vib, 'positive_cur': positive_cur,
#         #     'negative_vib': negative_vib, 'negative_cur': negative_cur,
#         #     'anchor_vib_fft': torch.from_numpy(anchor_vib_fft), 'anchor_cur_fft': torch.from_numpy(anchor_cur_fft),
#         #     'positive_vib_fft': torch.from_numpy(positive_vib_fft), 'positive_cur_fft': torch.from_numpy(positive_cur_fft),
#         #     'negative_vib_fft': torch.from_numpy(negative_vib_fft), 'negative_cur_fft': torch.from_numpy(negative_cur_fft)
#         # }