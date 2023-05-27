import torch
from torch import nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1x1conv:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.downsample:
            X = self.downsample(X)
        return F.relu(Y + X)
    

    
class block(nn.Module):
    def __init__(self,in_channels, out_channels,first_block=False):
        super(block,self).__init__()
        if first_block:
            assert in_channels == out_channels 
            
        if not first_block:
            self.residual0=Residual(in_channels, out_channels, use_1x1conv=True, stride=2)
        else:
            self.residual0=Residual(in_channels, out_channels)
                
        self.residual1 = Residual(out_channels, out_channels)
                    
    def forward(self,X):
        Y = self.residual0(X)
        Y = self.residual1(Y)
        return Y
        


# In[62]:


class resnet18(nn.Module):
    def __init__(self,num_class):
        super(resnet18,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        #self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = block(in_channels=64, out_channels=64,first_block=True)
        self.layer2 = block(64, 128)
        self.layer3 = block(128, 256)
        self.layer4 = block(256, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512,num_class)
            
    def forward(self,X):
        z = self.bn1(self.conv1(X))
        #z = self.pool(self.relu(z))
        z = self.relu(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)    
        z = self.global_avg_pool(z)
        z = torch.flatten(z,1)
        y = self.linear(z)
        return y