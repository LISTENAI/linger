import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import sys
import os
import torch.nn.functional
import linger
from linger.ops import *
import pytest
import numpy 
# Hyper Parameters
def test_quant_tensot_param():
    EPOCH = 5               # train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE = 50
    LR = 0.001              # learning rate
    DATA_DIR = '/yrfs2/bitbrain/data/'
    DOWNLOAD_MNIST = False
    TEST_SAMPLE = 1000

    torch.backends.cudnn.enabled = False
    device_id =0

    device = torch.device("cuda:"+str(device_id))
    torch.manual_seed(0)

    train_data = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,                                    
        transform=torchvision.transforms.ToTensor(), 
        download=DOWNLOAD_MNIST,
    )

    # plot one example
    #print(train_data.train_data.size())                 # (60000, 28, 28)
    #print(train_data.train_labels.size())               # (60000)


    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

    # pick 2000 samples to speed up testing
    test_data = torchvision.datasets.MNIST(root=DATA_DIR, train=False)
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:TEST_SAMPLE]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.targets[:TEST_SAMPLE]


    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(        
                Conv2dInt(
                    in_channels=1,             
                    out_channels=16,           
                    kernel_size=5,             
                    stride=1,                
                    padding=2,                 
                ),                             
                nn.ReLU(),                     
                nn.MaxPool2d(kernel_size=2),   
            )
            self.conv2 = nn.Sequential(        
                Conv2dInt(16, 32, 5, 1, 2),    
                nn.ReLU(),                      
                nn.MaxPool2d(2),                
            )
            self.out = LinearInt(32 * 7 * 7, 10)   


        def forward(self, x):        
            x = self.conv1(x)
            x = x*x
            
            x = linger.quant_tensor(self,x, name='_default_layername')
            layer = linger.quant_tensor_getlayer(self)
            if self.training:
                scale_local = 127/(x.abs().max())
            else:
                scale_local = layer.scale_x

            t = scale_local * x
            assert torch.sum(torch.frac(t+0.0001)).data  < 0.00011*t.numel()
            x = self.conv2(x)
            x = x.view(x.size(0), -1)           
            output = self.out(x)

            return output

    def test_net_forward_backward():
        cnn = CNN()
        cnn = cnn.cuda(device)
        #cnn.load_state_dict(torch.load('init.pt'))
    

        optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   
        loss_func = nn.CrossEntropyLoss()                      
        count_step = 0
        print("begin to train raw net ...")

        for epoch in range(1):
            for step, (b_xx, b_yy) in enumerate(train_loader):   
                cnn.train()
                b_x = b_xx.cuda(device)
                b_y = b_yy.cuda(device)
                output = cnn(b_x)
                loss = loss_func(output, b_y)   
                optimizer.zero_grad()          
                loss.backward()                
                optimizer.step()  
                # '''
                if (step+1) % 50 == 0:
                    cnn.eval()
                    accuracy = 0
                    for it in range(0,TEST_SAMPLE,BATCH_SIZE):
                        test_output = cnn(test_x[it:it+BATCH_SIZE].cuda(device))
                        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                        accuracy += float((pred_y == test_y[it:it+BATCH_SIZE].data.numpy()).astype(int).sum()) / float(BATCH_SIZE)
                    accuracy /= (TEST_SAMPLE/BATCH_SIZE)
                    print('Batch: ', step, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f ' % (100*accuracy))
                count_step+=1
        assert type(cnn._ifly_bitbrain_round_tensor_iq_tensor_quant__default_layername) == linger.ScaledRoundLayer

    # test_net_forward_backward()                   
