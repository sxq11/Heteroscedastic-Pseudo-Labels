import torch 
import torch.nn as nn
import torch.nn.functional as F
    

class UncertaintyLearner(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=1):
        super(UncertaintyLearner, self).__init__()
        self.fc = self.build_fc_net(input_dim, output_dim, hidden_dim, num_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def build_fc_net(self, input_dim, output_dim, hidden_dim, num_layers):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.fc(x)
        return out
    
    
