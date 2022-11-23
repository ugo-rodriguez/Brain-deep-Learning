# from sklearn.metrics import classification_report
# y_true = [0, 1, 2, 1, 1]
# y_pred = [0, 1, 2, 2, 1]
# target_names = ['class 0', 'class 1', 'class 2']
# print(classification_report(y_true, y_pred, target_names=target_names))

# from torch.nn.functional import softmax

# torch.tensor([2.0,1,0.1])

# import torch

# l_features = []

# path_data = "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness"
# number_brain = '101247'
# version = 'V06'
# hemishpere = 'left'

# path_eacsf = f"{path_data}/{number_brain}/{version}/eacsf/{hemishpere}_eacsf.txt"
# path_sa =    f"{path_data}/{number_brain}/{version}/sa/{hemishpere}_sa.txt"
# path_thickness = f"{path_data}/{number_brain}/{version}/thickness/{hemishpere}_thickness.txt"

# eacsf = open(path_eacsf,"r").read().splitlines()
# eacsf = torch.tensor([float(ele) for ele in eacsf])
# l_features.append(eacsf.unsqueeze(dim=1))

# sa = open(path_sa,"r").read().splitlines()
# sa = torch.tensor([float(ele) for ele in sa])
# l_features.append(sa.unsqueeze(dim=1))

# thickness = open(path_thickness,"r").read().splitlines()
# thickness = torch.tensor([float(ele) for ele in thickness])
# l_features.append(thickness.unsqueeze(dim=1))

# import plotly.express as px
# from sklearn.metrics import confusion_matrix
# y_true = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
# y_pred = [[0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [1], [1], [1], [1], [0], [0], [1], [0], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [1], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [1], [0], [1], [1], [0], [0], [1], [1], [0], [0], [0], [0], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [1], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0], [1], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [1], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [0], [1], [1], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [1], [1]]
# a = confusion_matrix(
#     y_true, y_pred
# )



# fig = px.imshow(a,labels=dict(x="Predicted condition", y="Actual condition"))
# fig.update_xaxes(side="top")
# fig.show()
            
# import torch
# from torch import nn

# m = nn.MaxPool1d(12)
# t = torch.randn(8, 12, 512)
# t = t.permute(0,2,1)
# output = m(t)
# output = output.squeeze(dim=2)

# import torch
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50

# model = resnet50(pretrained=True)
# target_layers = [model.layer4[-1]]
# input_tensor = torch.randn(18,3,7,7)
# cam = GradCAM(model=model, target_layers=target_layers)

# targets = [ClassifierOutputTarget(281)]

# # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# # In this example grayscale_cam has only one image in the batch:
# grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(input_tensor, grayscale_cam, use_rgb=True)

import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain-pytorch-lightning/Classification')

import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain-pytorch-lightning/Librairies')

import numpy as np
import cv2

import torch
from torch import nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

import monai
import pandas as pd

from brain_module_for_classification import BrainNet,BrainIcoNet, BrainIcoAttentionNet
from brain_data_for_classification import BrainIBISDataModule

from transformation import RandomRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterSphereTransform


import numpy as np
import random
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl 

from vtk.util.numpy_support import vtk_to_numpy
import vtk

import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight

import utils
from utils import ReadSurf, PolyDataToTensors

import pandas as pd


batch_size = 8
image_size = 224
noise_lvl = 0.03
dropout_lvl = 0.2
num_epochs = 100
ico_lvl = 1
radius=2
lr = 1e-5

mean = 0
std = 0.01

min_delta_early_stopping = 0.00
patience_early_stopping = 20

path_data = "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness"
train_path = "/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain-pytorch-lightning/Data/dataV06_V12_train2.csv"
val_path = "/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain-pytorch-lightning/Data/dataV06_V12_val2.csv"
test_path = "/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain-pytorch-lightning/Data/dataV06_V12_test2.csv"
path_ico = '/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk'

path_model = "Checkpoint/epoch=51-val_loss=0.11.ckpt"

list_train_transform = []    
list_train_transform.append(CenterSphereTransform())
list_train_transform.append(NormalizePointTransform())
list_train_transform.append(RandomRotationTransform())
list_train_transform.append(GaussianNoisePointTransform(mean,std))
list_train_transform.append(NormalizePointTransform())

train_transform = monai.transforms.Compose(list_train_transform)

list_val_and_test_transform = []    
list_val_and_test_transform.append(CenterSphereTransform())
list_val_and_test_transform.append(NormalizePointTransform())

val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

path_model = "Checkpoint/epoch=45-val_loss=0.14.ckpt"

brain_data = BrainIBISDataModule(batch_size,path_data,train_path,val_path,test_path,path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform)
nbr_features = brain_data.get_features()

model = BrainIcoAttentionNet(nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, radius=radius,lr=lr)
# checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
checkpoint = torch.load(path_model)
model.load_state_dict(checkpoint['state_dict'])

trainer = Trainer(max_epochs=num_epochs,accelerator="gpu")

trainer.test(model, datamodule=brain_data)


print(2)