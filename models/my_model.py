import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_classes=210):
        super().__init__()
        
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(3, 8, 7)
        self.c_act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 32, 3)
        self.c_act2 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.c_act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.c_act4 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.c_act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2)

        self.conv6 = nn.Conv2d(256, 256, 3)
        self.c_act6 = nn.ReLU()

        self.flattener = nn.Flatten()

        self.bn3 = nn.BatchNorm1d(9216)

        self.linear1 = nn.Linear(9216, 4096)
        self.l_act1 = nn.ReLU()

        self.linear2 = nn.Linear(4096, 1024)
        self.l_act2 = nn.ReLU()

        self.linear3 = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.pool1(self.c_act1(self.conv1(x)))
        
        x = self.c_act2(self.conv2(x))
        x = self.bn1(x)
        
        x = self.pool3(self.c_act3(self.conv3(x)))
        
        x = self.c_act4(self.conv4(x))
        x = self.bn2(x)
        
        x = self.pool5(self.c_act5(self.conv5(x)))
        
        x = self.c_act6(self.conv6(x))

        x = self.flattener(x)
        x = self.bn3(x)
        
        x = self.l_act1(self.linear1(x))
        
        x = self.l_act2(self.linear2(x))
        
        x = self.linear3(x)
        return x
        
