import torch
import torch.nn as nn
import torch.nn.functional as F

import YOLOLayer as yl

class BasicTestingModel(nn.Module):
    '''Module to be tested on torch.jit.script'''
    
    def __init__(self):
        super(BasicTestingModel, self).__init__()
        
        self.sequential_0 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        
        self.sequential_2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        
        self.sequential_4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_5 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        
        self.sequential_6 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_7 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        
        self.sequential_8 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_9 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        
        self.sequential_10 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn. BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_11 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False))
        
        self.sequential_12 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(1024, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_13 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                          nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_14 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_15 = nn.Sequential(nn.Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1)))
        
        self.sequential_16 = nn.Sequential(yl.YOLOLayer())
        
        self.sequential_17 = nn.Sequential(EmptyLayer())
        
        self.sequential_18 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                          nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_19 = nn.Sequential(Upsample())
        
        self.sequential_20 = nn.Sequential(EmptyLayer())
        
        self.sequential_21 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True),
                                         nn.LeakyReLU(negative_slope=0.1))
        
        self.sequential_22 = nn.Sequential(nn.Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1)))
        
        self.sequential_23 = nn.Sequential(yl.YOLOLayer())
        
        
        
        
        
        print("BasicTestingModel initiated.")
        
    def forward(self,x):
        x = self.sequential_1(x)
        x = self.sequential_2(x)
        x = self.sequential_3(x)
        x = self.sequential_4(x)
        x = self.sequential_5(x)
        x = self.sequential_6(x)
        x = self.sequential_7(x)
        x = self.sequential_8(x)
        x = self.sequential_9(x)
        x = self.sequential_10(x)
        x = self.sequential_11(x)
        x = self.sequential_12(x)
        x = self.sequential_13(x)
        x = self.sequential_14(x)
        x = self.sequential_15(x)
        x = self.sequential_16(x)
        x = self.sequential_17(x)
        x = self.sequential_18(x)
        x = self.sequential_19(x)
        x = self.sequential_20(x)
        x = self.sequential_21(x)
        x = self.sequential_22(x)
        x = self.sequential_23(x)
        
        return x
    
    

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

#     @torch.jit.script_method
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x