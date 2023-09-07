import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn import BasicBlock,GlobalAvgPool1D, conv3x1,conv1x1


class teacher_encoder(nn.Module):
    def __init__(self, block, layers, in_channel=2, out_channel=4, zero_init_residual=False):
        super(teacher_encoder, self).__init__()
        self.vib_fft_to_raw=nn.Linear(2048, 4096)
        self.vib_encoder_raw = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.vib_encoder_fft = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        
        self.cur_fft_to_raw=nn.Linear(2048, 4096)
        self.cur_encoder_raw = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.cur_encoder_fft = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        
        

    def forward(self, vib_raw, vib_fft, cur_raw, cur_fft):
        vib_fft = self.vib_fft_to_raw(vib_fft)
        vib_fft=self.vib_encoder_fft(vib_fft)
        
        vib_raw=self.vib_encoder_raw(vib_raw)
        
        cur_fft = self.cur_fft_to_raw(cur_fft)
        cur_fft=self.cur_encoder_fft(cur_fft)
        
        cur_raw=self.cur_encoder_raw(cur_raw)

        return vib_raw,cur_raw,vib_fft,cur_fft
    


    
class teacher_classifier(nn.Module):
    def __init__(self, block, layers, in_channel=2, out_channel=4, zero_init_residual=False):
        super(teacher_classifier, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=3, bias=False)
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

        return out, logits2,feature