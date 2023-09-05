class Domain_classifier1(nn.Module):

    def __init__(self):
        super(Domain_classifier1, self).__init__()
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
        logits1 = self.fc1(input)
        logits2 = self.fc2(logits1)
        return logits1, logits2
class Domain_classifier2(nn.Module):

    def __init__(self):
        super(Domain_classifier2, self).__init__()
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
        logits1 = self.fc1(input)
        logits2 = self.fc2(logits1)
        return logits1, logits2
    