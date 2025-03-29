#!/usr/bin/env python
# coding: utf-8

# ## 0 Preparation

# ### 0.1 Imports

# In[44]:


import torch
import random
import torch.nn as nn
from time import time_ns
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from torchvision.models import resnet152, resnet101
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
from datetime import datetime

import torch.nn.utils.prune as prune

from quanto import quantize, freeze, Calibration, qint8

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import rasterio 
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import os  
import cv2
import torch

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ### 0.2 Constants Definition

# In[21]:


BATCH_SIZE = 16
# SEED = random.randint(0,10000)
SEED = 42

def set_random_seed(seed=42):
    random.seed(seed) # set python seed
    np.random.seed(seed) # seed the global NumPy random number generator(RNG)
    torch.manual_seed(seed) # seed the RNG for all devices(both CPU and CUDA) 

set_random_seed(seed=SEED)

# base folder (change needed)
base_path = "/mnt/AI4EO-MapYourCity/v1/building-age-dataset/" # This line has to be modified/ changed  
train_path = base_path + "train/data/"
test_path =  base_path + "test/data/"

train_data_names = os.listdir(train_path)
test_data_names=os.listdir(test_path)

print(len(train_data_names))

#make validation dataset
n=len(train_data_names)*10//100
train_data_names, val_data_names= torch.utils.data.random_split(train_data_names, [n, len(train_data_names) - n])

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


# ### 0.3 Dataset Definition

# In[22]:


class MYCDataset(torch.utils.data.Dataset):    
    """
    This class defines the data with all the 3 modalities   
    """
    def __init__(self, list_IDs,transform=None,train=True):
        """
        This function initializes the data class - constructor function   
        :param list_IDs: the PID numbers - (i.e. the pid) 
        """
        self.list_IDs = list_IDs 
        self.transform=transform
        self.train=train
        self.path=train_path if train else test_path

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index): 
        ID = self.list_IDs[index] 
        exists=os.path.exists(self.path + ID + "/street.jpg")
        if exists:
            X = cv2.imread(self.path + ID + '/street.jpg')
            X = cv2.resize(X,(64,64))
            X = np.transpose(X,[2,0,1])

        
        
        with rasterio.open(self.path + ID + '/orthophoto.tif') as src:
            # resample data to target shape
            X2 = src.read(
                out_shape=(
                    src.count,
                    256,
                    256
                ),
                resampling=Resampling.bilinear
            )


            #X2 = np.transpose(X2,(1,0,2))

        with rasterio.open(self.path + ID + '/s2_l2a.tif') as src:
            # resample data to target shape
            X3 = src.read(
                out_shape=(
                    src.count,
                    128,
                    128
                ),
                resampling=Resampling.bilinear
            )

            #X3 = np.transpose(X3,(1,0,2))
        
            
        # X2 = rasterio.open(self.path + ID + '/orthophoto.tif').read()
        
        # X3 = rasterio.open(self.path + ID + '/s2_l2a.tif').read() 
        
        if self.train:
            y = int(open(self.path + ID + '/label.txt', "r").read())
            
        if self.train:
            return self.transform(X,X2,X3,y) if self.transform else (X,X2,X3,y)
        if exists and not self.train:
            return self.transform(X,X2,X3) if self.transform else (X,X2,X3)
        return self.transform(X2,X3) if self.transform else (X2,X3)


# ### 0.4 Transform definition

# In[23]:


class MyTransform:
    def __init__(self, transform_X1, transform_X2, transform_X3):
        self.transform_X1 = transform_X1
        self.transform_X2 = transform_X2
        self.transform_X3 = transform_X3

    def __call__(self, *args):
        num_args=len(args)
        if num_args==4:
            return (self.transform_X1(args[0]),\
                self.transform_X2(args[1]).permute(1, 0, 2),\
                self.transform_X3(args[2]).permute(1, 0, 2),\
                args[3])
        if num_args==3:
            return (self.transform_X1(args[0]).permute(1,0,2),\
                self.transform_X2(args[1]).permute(1, 0, 2),\
                self.transform_X3(args[2]).permute(1, 0, 2))
        return (torch.zeros(3,64,64),\
            self.transform_X2(args[0]).permute(1, 0, 2),\
            self.transform_X3(args[1]).permute(1, 0, 2))


# In[24]:


transform_x1=transforms.Compose([
])
transform_x2=transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15)
])
transform_x3=transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15)
])

transform=MyTransform(transform_x1,transform_x2,transform_x3)

transform_x1_test=transforms.Compose([
    transforms.ToTensor()
])
transform_x2_test=transforms.Compose([
    transforms.ToTensor(),
])
transform_x3_test=transforms.Compose([
    transforms.ToTensor(),
])
transform_test=MyTransform(transform_x1_test,transform_x2_test,transform_x3_test)


# ### 0.6 Model Definition

# In[25]:


class SE_Block(nn.Module): 
    def __init__(self, channels, reduction=16, activation="relu"):
        super().__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU6(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.ReLU6):
        return activation_name

    elif activation_name == "gelu":
        return nn.GELU()
    elif isinstance(activation_name, torch.nn.modules.activation.GELU):
        return activation_name

    elif activation_name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.LeakyReLU):
        return activation_name

    elif activation_name == "prelu":
        return nn.PReLU()
    elif isinstance(activation_name, torch.nn.modules.activation.PReLU):
        return activation_name

    elif activation_name == "selu":
        return nn.SELU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.SELU):
        return activation_name

    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif isinstance(activation_name, torch.nn.modules.activation.Sigmoid):
        return activation_name

    elif activation_name == "tanh":
        return nn.Tanh()
    elif isinstance(activation_name, torch.nn.modules.activation.Tanh):
        return activation_name

    elif activation_name == "mish":
        return nn.Mish()
    elif isinstance(activation_name, torch.nn.modules.activation.Mish):
        return activation_name
    else:
        raise ValueError(f"activation must be one of leaky_relu, prelu, selu, gelu, sigmoid, tanh, relu. Got: {activation_name}")

def get_normalization(normalization_name, num_channels, num_groups=32, dims=2):
    if normalization_name == "batch":
        if dims == 1:
            return nn.BatchNorm1d(num_channels)
        elif dims == 2:
            return nn.BatchNorm2d(num_channels)
        elif dims == 3:
            return nn.BatchNorm3d(num_channels)
    elif normalization_name == "instance":
        if dims == 1:
            return nn.InstanceNorm1d(num_channels)
        elif dims == 2:
            return nn.InstanceNorm2d(num_channels)
        elif dims == 3:
            return nn.InstanceNorm3d(num_channels)
    elif normalization_name == "layer":
        return nn.LayerNorm(num_channels)
    elif normalization_name == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normalization_name == "bcn":
        if dims == 1:
            return nn.Sequential(
                nn.BatchNorm1d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 2:
            return nn.Sequential(
                nn.BatchNorm2d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 3:
            return nn.Sequential(
                nn.BatchNorm3d(num_channels),
                nn.GroupNorm(1, num_channels)
            )    
    elif normalization_name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"normalization must be one of batch, instance, layer, group, none. Got: {normalization_name}") 


# In[26]:


class CoreCNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, *, norm="batch", activation="relu", padding="same", residual=True):

        super(CoreCNNBlock, self).__init__()



        self.activation = get_activation(activation)

        self.residual = residual

        self.padding = padding

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.squeeze = SE_Block(self.out_channels)



        self.match_channels = nn.Identity()

        if in_channels != out_channels:

            self.match_channels = nn.Sequential(

                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),

                get_normalization(norm, out_channels),

            )



        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0)

        self.norm1 = get_normalization(norm, self.out_channels)



        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=self.out_channels)

        self.norm2 = get_normalization(norm, self.out_channels)

        

        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=1)

        self.norm3 = get_normalization(norm, self.out_channels)



    def forward(self, x):

        identity = x

        x = self.activation(self.norm1(self.conv1(x)))

        x = self.activation(self.norm2(self.conv2(x)))

        x = self.norm3(self.conv3(x))

        x = x * self.squeeze(x)

        if self.residual:

            x = x + self.match_channels(identity)

        x = self.activation(x) 

        return x



class CoreEncoderBlock(nn.Module): 

    def __init__(self, depth, in_channels, out_channels, norm="batch", activation="relu", padding="same"):

        super(CoreEncoderBlock, self).__init__() 

        self.depth = depth

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.activation = activation

        self.norm = norm

        self.padding = padding

        self.blocks = []

        for i in range(self.depth): 

            _in_channels = self.in_channels if i == 0 else self.out_channels

            block = CoreCNNBlock(_in_channels, self.out_channels, norm=self.norm, activation=self.activation, padding=self.padding)



            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    

    def forward(self, x):

        for i in range(self.depth):

            x = self.blocks[i](x)

        before_downsample = x

        x = self.downsample(x)

        return x, before_downsample



class CoreAttentionBlock(nn.Module):

    def __init__(self,

        lower_channels,

        higher_channels, *,

        norm="batch",

        activation="relu",

        padding="same",

    ):

        super(CoreAttentionBlock, self).__init__()

        self.lower_channels = lower_channels

        self.higher_channels = higher_channels

        self.activation = get_activation(activation)

        self.norm = norm

        self.padding = padding

        self.expansion = 4

        self.reduction = 4

        if self.lower_channels != self.higher_channels:

            self.match = nn.Sequential(

                nn.Conv2d(self.higher_channels, self.lower_channels, kernel_size=1, padding=0, bias=False),

                get_normalization(self.norm, self.lower_channels),

            )

        self.compress = nn.Conv2d(self.lower_channels, 1, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

        self.attn_c_pool = nn.AdaptiveAvgPool2d(self.reduction)

        self.attn_c_reduction = nn.Linear(self.lower_channels * (self.reduction ** 2), self.lower_channels * self.expansion)

        self.attn_c_extention = nn.Linear(self.lower_channels * self.expansion, self.lower_channels)



    def forward(self, x, skip):

        if x.size(1) != skip.size(1):

            x = self.match(x)

        x = x + skip

        x = self.activation(x)

        attn_spatial = self.compress(x)

        attn_spatial = self.sigmoid(attn_spatial)

        attn_channel = self.attn_c_pool(x)

        attn_channel = attn_channel.reshape(attn_channel.size(0), -1)

        attn_channel = self.attn_c_reduction(attn_channel)

        attn_channel = self.activation(attn_channel)

        attn_channel = self.attn_c_extention(attn_channel)

        attn_channel = attn_channel.reshape(x.size(0), x.size(1), 1, 1)

        attn_channel = self.sigmoid(attn_channel)

        return attn_spatial, attn_channel



class CoreDecoderBlock(nn.Module):

    def __init__(self, depth, in_channels, out_channels, *, norm="batch", activation="relu", padding="same"):

        super(CoreDecoderBlock, self).__init__()

        self.depth = depth

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.activation_blocks = activation

        self.activation = get_activation(activation)

        self.norm = norm

        self.padding = padding

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.match_channels = CoreCNNBlock(self.in_channels * 2, self.out_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)

        self.attention = CoreAttentionBlock(self.in_channels, self.in_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)

        self.blocks = []

        for _ in range(self.depth):

            block = CoreCNNBlock(self.out_channels, self.out_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)

            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)

    

    def forward(self, x, skip):

        x = self.upsample(x)

        attn_s, attn_c = self.attention(x, skip)

        x = torch.cat([x, (skip * attn_s) + (skip + attn_c)], dim=1)

        x = self.match_channels(x)

        for i in range(self.depth):

            x = self.blocks[i](x)

        return x



class CoreUnet(nn.Module):  

    def __init__(self, *,

        input_dim=10,

        output_dim=1,

        depths=None,

        dims=None,

        activation="relu",

        norm="batch",

        padding="same",

    ): 

        super(CoreUnet, self).__init__() 

        self.depths = [3, 3, 9, 3] if depths is None else depths 

        self.dims = [96, 192, 384, 768] if dims is None else dims

        #self.depths = [3, 3, 9] if depths is None else depths

        #self.dims = [96, 192, 384] if dims is None else dims

        self.output_dim = output_dim

        self.input_dim = input_dim

        self.activation = activation

        self.norm = norm

        self.padding = padding

        self.dims = [v // 2 for v in self.dims] 

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length. "   

        self.stem = nn.Sequential(

            CoreCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),

        )  

        self.encoder_blocks = []  

        for i in range(len(self.depths)):

            encoder_block = CoreEncoderBlock(

                self.depths[i],

                self.dims[i - 1] if i > 0 else self.dims[0],

                self.dims[i],

                norm=self.norm,

                activation=self.activation,

                padding=self.padding,

            )

            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.decoder_blocks = [] 

        for i in reversed(range(len(self.encoder_blocks))):

            decoder_block = CoreDecoderBlock(

                self.depths[i],

                self.dims[i],

                self.dims[i - 1] if i > 0 else self.dims[0],

                norm=self.norm,

                activation=self.activation,

                padding=self.padding,

            )

            self.decoder_blocks.append(decoder_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.bridge = nn.Sequential(

            CoreCNNBlock(self.dims[-1], self.dims[-1], norm=self.norm, activation=self.activation, padding=self.padding),

        )

        self.head = nn.Sequential(

            CoreCNNBlock(self.dims[0], self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),

            nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),

        )



    def forward(self, x):

        skip_connections = []    

        x = self.stem(x)

        for block in self.encoder_blocks:

            x, skip = block(x)

            skip_connections.append(skip)

        x = self.bridge(x)

        return x



class CoreEncoder(nn.Module):

    def __init__(self, *,

        input_dim=10,

        output_dim=1,

        depths=None,

        dims=None,

        activation="relu",

        norm="batch",

        padding="same",

    ):

        super(CoreEncoder, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths

        self.dims = [96, 192, 384, 768] if dims is None else dims

        self.output_dim = output_dim

        self.input_dim = input_dim

        self.activation = activation

        self.norm = norm

        self.padding = padding

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = CoreCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding)

        self.encoder_blocks = []  

        for i in range(len(self.depths)): 

            encoder_block = CoreEncoderBlock(

                self.depths[i],

                self.dims[i - 1] if i > 0 else self.dims[0],

                self.dims[i],

                norm=self.norm,

                activation=self.activation,

                padding=self.padding,

            )

            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.head = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(self.dims[-1], self.output_dim),

        )



    def forward(self, x):

        x = self.stem(x)

        for block in self.encoder_blocks:

            x, _ = block(x)

        x = self.head(x)

        return x



class ResNet152(nn.Module):

    def __init__(self, pretrained):

        super(ResNet152, self).__init__() 

        #self.model = pretrainedmodels.__dict__['resnet152'](pretrained='imagenet')                         

        #self.model = torchvision.models.resnet152(pretrained=True)                          

        class MyResNet18(nn.Module):

            def __init__(self, resnet, resnet2):

                super().__init__()

                self.features = nn.Sequential(

                    resnet.conv1,

                    resnet.bn1,

                    resnet.relu,

                    resnet.maxpool,

                    resnet.layer1,

                    resnet.layer2,

                    resnet.layer3,

                    resnet.layer4

                ) 

                self.avgpool = resnet.avgpool

                self.fc = resnet.fc



                self.features2 = nn.Sequential(

                    resnet2.conv1,

                    resnet2.bn1,

                    resnet2.relu,

                    resnet2.maxpool,

                    resnet2.layer1,

                    resnet2.layer2,

                    resnet2.layer3,

                    resnet2.layer4

                )

                self.avgpool2 = resnet2.avgpool

                self.fc2 = resnet2.fc



            def _forward_impl(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

                x = self.features(x)

                x = self.avgpool(x)

                x = torch.flatten(x, 1)

                x = self.fc(x)

                x2 = self.features2(x2)

                x2 = self.avgpool2(x2)

                x2 = torch.flatten(x2, 1)

                x2 = self.fc2(x2) 

                return x, x2



            def forward(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

                return self._forward_impl(x, x2) 



        model = resnet152(weights="DEFAULT")

        model2 = resnet152(weights="DEFAULT")

        self.model = MyResNet18(model, model2)

        self.l0 = nn.Linear(4480, 7)



    def forward(self, x1, x2, x3):

        batch, _, _, _ = x1.shape        

        CHANNELS = 12

        model = CoreUnet(

            input_dim=CHANNELS,

            output_dim=1,

        ).to(device)   

        x2 = self.model.features2(x2)
        x3 = model(x3) 

        x2 = F.adaptive_avg_pool2d(x2, 1).reshape(batch, -1) 
        x3 = F.adaptive_avg_pool2d(x3, 1).reshape(batch, -1) 

        if torch.any(x1 != torch.zeros_like(x1)):        
            x1 = self.model.features(x1)
            x1 = F.adaptive_avg_pool2d(x1, 1).reshape(batch, -1)   
            x = torch.cat((x1, x2, x3), 1)
        else:
            x = torch.cat((x2,x3),1)

        l0 = self.l0(x)

        return l0 
    


class ResNet101(nn.Module):

    def __init__(self, pretrained):

        super(ResNet101, self).__init__() 

        #self.model = pretrainedmodels.__dict__['resnet152'](pretrained='imagenet')                         

        #self.model = torchvision.models.resnet152(pretrained=True)                          

        class MyResNet18(nn.Module):

            def __init__(self, resnet, resnet2):

                super().__init__()

                self.features = nn.Sequential(

                    resnet.conv1,

                    resnet.bn1,

                    resnet.relu,

                    resnet.maxpool,

                    resnet.layer1,

                    resnet.layer2,

                    resnet.layer3,

                    resnet.layer4

                ) 

                self.avgpool = resnet.avgpool

                self.fc = resnet.fc



                self.features2 = nn.Sequential(

                    resnet2.conv1,

                    resnet2.bn1,

                    resnet2.relu,

                    resnet2.maxpool,

                    resnet2.layer1,

                    resnet2.layer2,

                    resnet2.layer3,

                    resnet2.layer4

                )

                self.avgpool2 = resnet2.avgpool

                self.fc2 = resnet2.fc



            def _forward_impl(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

                x = self.features(x)

                x = self.avgpool(x)

                x = torch.flatten(x, 1)

                x = self.fc(x)

                x2 = self.features2(x2)

                x2 = self.avgpool2(x2)

                x2 = torch.flatten(x2, 1)

                x2 = self.fc2(x2) 

                return x, x2



            def forward(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

                return self._forward_impl(x, x2) 



        model = resnet101(weights="DEFAULT")

        model2 = resnet101(weights="DEFAULT")

        self.model = MyResNet18(model, model2)

        self.l0 = nn.Linear(4480, 7)



    def forward(self, x1, x2, x3):

        batch, _, _, _ = x1.shape        

        CHANNELS = 12

        model = CoreUnet(

            input_dim=CHANNELS,

            output_dim=1,

        ).to(device)   

        x2 = self.model.features2(x2)
        x3 = model(x3) 

        x2 = F.adaptive_avg_pool2d(x2, 1).reshape(batch, -1) 
        x3 = F.adaptive_avg_pool2d(x3, 1).reshape(batch, -1) 

        if torch.any(x1 != torch.zeros_like(x1)):        
            x1 = self.model.features(x1)
            x1 = F.adaptive_avg_pool2d(x1, 1).reshape(batch, -1)   
            x = torch.cat((x1, x2, x3), 1)
        else:
            x = torch.cat((x2,x3),1)

        l0 = self.l0(x)

        return l0 


# ## 1 Dataloader instantiation

# ### 1.1 Read csv files

# In[27]:


test_df = pd.read_csv(base_path + "test/test-set.csv")
train_df = pd.read_csv(base_path + "train/train-set.csv")


# 

# ### 1.2 Dataloader instantation

# In[28]:


train_set = MYCDataset(train_data_names,transform=transform)
val_set = MYCDataset(val_data_names,transform=transform) 
test_set  = MYCDataset(test_data_names,transform=transform_test,train=False)

train_dataloader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
val_dataloader = DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = DataLoader(test_set,batch_size=BATCH_SIZE*2,shuffle=False)


# ## 2.0 Training

# ### 2.1 Model

# In[29]:


model152_path = "./model152"
model152 = ResNet152(pretrained=True).to(device) 
model152.train()

model101_path = "./model101"
model101 = ResNet101(pretrained=True).to(device) 
model101.train()


# ### 2.2 Criterion and Optimizer

# In[30]:


criterion = nn.CrossEntropyLoss()


# ### 2.3 Eval Loop Definition

# In[31]:


def eval_loop(model,dataloader):
    start=time_ns()
    running_tloss = 0.
    test_acc = 0.
    num_tcorrect = 0
    num_tsamples = 0
    with torch.no_grad():
        for _, tdata in enumerate(dataloader):
            x1,x2,x3, tlabels = tdata
            x1=x1.to(device,dtype=torch.float32)
            x2=x2.to(device,dtype=torch.float32)
            x3=x3.to(device,dtype=torch.float32)  
            toutputs = model(x1,x2,x3)
            tloss = criterion(toutputs, tlabels.to(device))
            running_tloss += tloss
            _, tpredictions = toutputs.max(dim=-1)
            num_tcorrect += (tpredictions == tlabels.to(device)).sum()
            num_tsamples += tpredictions.size(0)
    avg_tloss = running_tloss/len(dataloader)
    test_acc = float(num_tcorrect)/float(num_tsamples)
    end=time_ns()
    return avg_tloss,test_acc, end-start


# ### 2.4 Train Loop Definition

# In[32]:


def train_one_epoch(model, optimizer):
    running_loss = 0.
    train_loss = 0.
    train_acc = 0.
    num_correct = 0
    num_samples = 0

    # Here, we use enumerate(train_dataloader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for _, data in enumerate(train_dataloader):
        
        # Every data instance is an input + label pair
        x1,x2,x3,labels = data        
        
        x1=x1.to(device,dtype=torch.float32)
        x2=x2.to(device,dtype=torch.float32)
        x3=x3.to(device,dtype=torch.float32)     


        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(x1,x2,x3)
        
        # Compute the loss and its gradients
        loss = criterion(outputs, labels.to(device))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        _, predictions = outputs.max(dim=1)
        num_correct += (predictions == labels.to(device)).sum()
        num_samples += predictions.size(0)

    train_loss = running_loss/len(train_dataloader)
    train_acc = float(num_correct)/float(num_samples)

    return train_loss, train_acc


# In[33]:


# Initializing in a separate cell so we can easily add more epochs to the same run
def train(model,path):
        # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('./eurosat_trainer_{}'.format(timestamp))
    epoch_number = 0
    tr_acc = 0.0
    best_vloss = 1_000_000.

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=5e-4)


    for epoch in range(25):
        print('============= EPOCH {} ============='.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, tr_acc = train_one_epoch(model,optimizer)

        running_vloss = 0.0
        val_acc = 0.0
        num_vcorrect = 0
        num_vsamples = 0

        # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                y1,y2,y3,vlabels = vdata
                y1=y1.to(device,dtype=torch.float32)
                y2=y2.to(device,dtype=torch.float32)
                y3=y3.to(device,dtype=torch.float32)     
                voutputs = model(y1,y2,y3)
                vloss = criterion(voutputs, vlabels.to(device))
                running_vloss += vloss
                _, vpredictions = voutputs.max(dim=-1)
                num_vcorrect += (vpredictions == vlabels.to(device)).sum()
                num_vsamples += vpredictions.size(0)

        avg_vloss = running_vloss / len(val_dataloader)
        val_acc = float(num_vcorrect)/float(num_vsamples)
        print('LOSS : train {} | valid {}'.format(round(avg_loss, 4), round(avg_vloss.item(), 4)))
        print('ACC  : train {}% | valid {}%'.format(round(tr_acc*100, 2), round(val_acc*100),2))

        # Log the running loss averaged per epoch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.add_scalars('Training vs. Validation Accuracy',
                        { 'Training' : tr_acc, 'Validation' : val_acc },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), path)

        epoch_number += 1


# In[34]:


def train_knowledge_distillation(teacher, student, T, soft_target_loss_weight, ce_loss_weight, device,student_path):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2, weight_decay=5e-4)

    teacher.eval()  # Teacher set to evaluation mode

    tr_acc = 0.0
    best_vloss = 1_000_000.

    for epoch in range(25):
        student.train() # Student to train mode

        print('============= EPOCH {} ============='.format(epoch + 1))

        running_loss = 0.
        num_correct = 0
        num_samples = 0
        running_vloss = 0.0
        val_acc = 0.0
        num_vcorrect = 0
        num_vsamples = 0

        for _, data in enumerate(train_dataloader):
            x1,x2,x3, labels = data
            x1=x1.to(device,dtype=torch.float32)
            x2=x2.to(device,dtype=torch.float32)
            x3=x3.to(device,dtype=torch.float32)
            labels=labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(x1,x2,x3)

            student_logits = student(x1,x2,x3)

            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predictions = student_logits.max(dim=-1)

            num_correct += (predictions == labels.to(device)).sum()
            num_samples += predictions.size(0)

        avg_loss = running_loss/len(train_dataloader)
        tr_acc = float(num_correct)/float(num_samples)
            # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                y1,y2,y3,vlabels = vdata
                y1=y1.to(device,dtype=torch.float32)
                y2=y2.to(device,dtype=torch.float32)
                y3=y3.to(device,dtype=torch.float32)     
                voutputs = student(y1,y2,y3)
                vloss = criterion(voutputs, vlabels.to(device))
                running_vloss += vloss
                _, vpredictions = voutputs.max(dim=-1)
                num_vcorrect += (vpredictions == vlabels.to(device)).sum()
                num_vsamples += vpredictions.size(0)

        avg_vloss = running_vloss / len(val_dataloader)
        val_acc = float(num_vcorrect)/float(num_vsamples)
        print('LOSS : train {} | valid {}'.format(round(avg_loss, 4), round(avg_vloss.item(), 4)))
        print('ACC  : train {}% | valid {}%'.format(round(tr_acc*100, 2), round(val_acc*100),2))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(student.state_dict(), student_path)


# In[35]:


train(model152, model152_path)
train(model101, model101_path)


# In[36]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def size(path):
    return os.path.getsize(path) / 1024**2


# In[37]:


def predictions(model,dataloader,path):
    res_df = test_df.copy()
    res_df["predicted_label"]=np.zeros(res_df.shape[0])
    i=0
    start = time_ns()
    with torch.no_grad():
       for _,tdata in enumerate(dataloader):
           x1,x2,x3 = tdata
           x1=x1.to(device,dtype=torch.float32)
           x2=x2.to(device,dtype=torch.float32)
           x3=x3.to(device,dtype=torch.float32)  
           toutputs = model(x1,x2,x3)
           _,tpredictions=toutputs.max(dim=-1)
           j = i + len(tpredictions)
           res_df.loc[i:j-1, "predicted_label"] = tpredictions.cpu().numpy()
           i = j
    time = time_ns() - start
    res_df["predicted_label"]=res_df["predicted_label"].astype("int8")
    res_df.to_csv(path)
    return time


# ## Baseline

# In[38]:


num_classes = 10
baseline_model = ResNet152(pretrained=True).to(device) 
baseline_model.load_state_dict(torch.load(model152_path))
baseline_model.to(device)


baseline_time = predictions(baseline_model,test_dataloader ,"baseline_submission.csv")
baseline_size = size(model152_path)
baseline_parameters = count_parameters(baseline_model)


# ## Pruning

# In[39]:


pruned_model=ResNet152(pretrained=True)
pruned_model.load_state_dict(torch.load(model152_path))
pruned_model.to(device)
pruned_model.eval()

pruned_path = "./pruned_model"

parameters_to_prune = []
for module_name, module in pruned_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        parameters_to_prune.append((module, "weight"))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4
)

for module,name in parameters_to_prune:
    prune.remove(module,name)

torch.save(pruned_model.state_dict(), pruned_path)

pruned_time = predictions(pruned_model, test_dataloader,"pruned_submission.csv")
pruned_size = size(pruned_path)
pruned_parameters = count_parameters(pruned_model)

print(f"The pruned model:") 
print(f" - has {(pruned_parameters/baseline_parameters)*100:.2f}% of parameters")
print(f" - is {baseline_size/pruned_size:.2f} times smaller")
print(f" - is {baseline_time/pruned_time:.2f} times faster")


# ## Distillation

# In[40]:


teacher_model=ResNet152(pretrained=True)
teacher_model.load_state_dict(torch.load(model152_path))
teacher_model.to(device)
teacher_model.eval()

small_model=ResNet101(pretrained=True)
small_model.load_state_dict(torch.load(model101_path))
small_model.to(device)
small_model.eval()

student_path = "./student_model"
student_model=ResNet101(pretrained=True)
student_model.to(device)
student_model.eval()

train_knowledge_distillation(teacher=teacher_model, student=student_model, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device,student_path=student_path)
# student_model.load_state_dict(torch.load(student_path))

small_time = predictions(small_model, test_dataloader,"small_submission.csv")
small_size = size(model101_path)
small_parameters = count_parameters(small_model)
print(f"The small model:") 
print(f" - has {(small_parameters/baseline_parameters)*100:.2f}% of parameters")
print(f" - is {baseline_size/small_size:.2f} times smaller")
print(f" - is {baseline_time/small_time:.2f} times faster")

student_time = predictions(student_model, test_dataloader, "student_submission.csv")
student_size = size(student_path)
student_parameters = count_parameters(student_model)
print(f"The student model:") 
print(f" - has {(student_parameters/baseline_parameters)*100:.2f}% of parameters")
print(f" - is {baseline_size/student_size:.2f} times smaller")
print(f" - is {baseline_time/student_time:.2f} times faster")


# ## Quantization

# In[42]:


def cal(model,dataloader):
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            x1,x2,x3 = data[0].to(device), data[1].to(device), data[2].to(device)
            model(x1,x2,x3)


# In[45]:


# quantized_path = "./quantized_model"
# quantized_model = ResNet152(pretrained = True)
# quantized_model.load_state_dict(torch.load(model152_path))
# quantized_model.to(device)
# quantized_model.eval()

# quantize(quantized_model, weights=qint8, activations=qint8)

# with Calibration():
#     cal(quantized_model, train_dataloader)

# freeze(quantized_model)

# torch.save(quantized_model.state_dict(), quantized_path)


# In[ ]:


# quantized_time = predictions(quantized_model,test_dataloader, "quantized_submission.csv")
# quantized_size = size(quantized_path)
# quantized_parameters = count_parameters(quantized_model)
# print(f"The quantized model:") 
# print(f" - has {(quantized_parameters/baseline_parameters)*100:.2f}% of parameters")
# print(f" - is {baseline_size/quantized_size:.2f} times smaller")
# print(f" - is {baseline_time/quantized_time:.2f} times faster")

