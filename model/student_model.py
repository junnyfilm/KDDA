import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from .cnn import BasicBlock,GlobalAvgPool1D, conv3x1,conv1x1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)
    
    
class student_encoder(nn.Module):
    def __init__(self, block, layers, in_channel=2, out_channel=4, zero_init_residual=False):
        super(student_encoder, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv1d(64, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
       
        self.gap = GlobalAvgPool1D()
   
        self.fft_to_raw=nn.Linear(2048, 4096)
        self.encoder_raw = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.encoder_fft = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, raw, fft):
        # x=torch.squeeze(x,3)
        fft = self.fft_to_raw(fft)
        # print("After fft_to_raw: ", torch.isnan(fft).any())
        # print("After fft_to_raw: ", fft)
    
        fft = self.encoder_fft(fft)
        # print("After encoder_fft: ", torch.isnan(fft).any())
        # print("After fft_to_raw: ", fft)
        raw = self.encoder_raw(raw)
        # print("After encoder_raw: ", torch.isnan(raw).any())
        # print("After fft_to_raw: ", raw)
        feature=torch.cat([raw,fft], dim=1)
        # print(feature)
       

        return raw,fft,feature
    
    
class student_classifier(nn.Module):
    def __init__(self, block, layers, in_channel=2, out_channel=4, zero_init_residual=False):
        super(student_classifier, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv1d(64, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.gap = GlobalAvgPool1D()
        
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(128, 5),
            nn.BatchNorm1d(5),
            nn.ReLU())
       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, feature):
        # x=torch.squeeze(x,3)
        
        x = self.conv1(feature)
        x = self.bn1(x)
        x = self.relu(x)
        feature = self.maxpool(x)

        x1 = self.layer1(feature)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.gap(x4)
        x = x.view(x.size(0),-1)
        logits = self.fc1(x)
        logits2 = self.fc2(logits)
        out = self.fc3(logits2)

        return out, logits2, feature,x
    
    
class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.fc0 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(32, 2),
            nn.BatchNorm1d(2),
            nn.ReLU())
        

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        logits0= self.fc0(input)
        logits1 = self.fc1(logits0)
        logits2 = self.fc2(logits1)
        return logits1, logits2
    
    

    
