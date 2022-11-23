import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain-pytorch-lightning/Librairies')

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision.models as models
from torch.nn.functional import softmax
import torchmetrics

import utils
from utils import ReadSurf, PolyDataToTensors, CreateIcosahedron

import layers
from layers import GaussianNoise, SelfAttention,Identity, TimeDistributed,IcosahedronConv2d

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRendererWithFragments, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.vis.plotly_vis import plot_scene

import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



class BrainIcoAttentionNet(pl.LightningModule):
    def __init__(self,nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, radius=1.35, lr=1e-4):
        super().__init__()

        self.save_hyperparameters()

        self.nbr_features = nbr_features
        self.dropout_lvl = dropout_lvl
        self.image_size = image_size
        self.noise_lvl = noise_lvl 
        self.batch_size = batch_size
        self.radius = radius

        ico_sphere = utils.CreateIcosahedron(self.radius, ico_lvl)
        ico_sphere_verts, ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = np.array(self.ico_sphere_edges)
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            #if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])): 
            #    R = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)    
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)
        self.R=torch.cat(R)
        self.T=torch.cat(T)

        efficient_net = models.resnet18()
        #efficient_net = models.efficientnet_b0()
        #efficient_net.features[0][0] = nn.Conv2d(self.nbr_features, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        efficient_net.classifier = Identity()

        self.drop = nn.Dropout(p=self.dropout_lvl)

        self.noise = GaussianNoise(mean=0.0, std=noise_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)

        conv2d = nn.Conv2d(1000, 512, kernel_size=(3,3),stride=2,padding=0) 
        self.IcosahedronConv2d = IcosahedronConv2d(conv2d,self.ico_sphere_verts,self.ico_sphere_edges)

        self.Attention = SelfAttention(1000, 128)
        self.Classification = nn.Linear(512, 2)

        self.Sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()




        # Initialize a perspective camera.
        self.cameras = FoVPerspectiveCameras()

        
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000
        )
        # We can add a point light in front of the object. 

        lights = AmbientLights()
        rasterizer = MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            )
        self.phong_renderer = MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.cameras, lights=lights)
        )
 
    def forward(self, x):

        V, F, VF, FF = x

        V = V.to(self.device)
        F = F.to(self.device)
        VF = VF.to(self.device)
        FF = FF.to(self.device)

        x, PF = self.render(V,F,VF,FF)

        x = self.noise(x)
        query = self.TimeDistributed(x)
        values = self.IcosahedronConv2d(query)

        x_a, w_a = self.Attention(query, values)
        x_a = self.drop(x_a)
        x = self.Classification(x_a)

        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def render(self,V,F,VF,FF):

        textures = TexturesVertex(verts_features=VF)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )

        # To plot the sphere
        # fig = plot_scene({"subplot1":{"meshes":meshes}})
        # fig.show()

        PF = []
        #for i in range(2):  # multiple views of the object
        for i in range(len(self.R)): 

            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))

        PF = torch.cat(PF, dim=1)
        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,:,index],PF)*(PF >= 0)) # take each feature
        x = torch.cat(l_features,dim=2)

        return x, PF

    def training_step(self, train_batch, batch_idx):

        V, F, VF, FF, Y = train_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        loss = self.loss(x,Y) 

        self.log('train_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.batch_size)  

        return loss

    def validation_step(self,val_batch,batch_idx):
        
        V, F, VF, FF, Y = val_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        #Last loss : for BCE lines
        # x = self.Sigmoid(x).squeeze(dim=1)
        #loss = self.loss(x,Y.to(torch.float32)) 

        loss = self.loss(x,Y) 

        self.log('val_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.val_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("val_acc", self.val_accuracy, batch_size=self.batch_size) 


    def test_step(self,test_batch,batch_idx):
        
        V, F, VF, FF, Y = test_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        loss = self.loss(x,Y) 

        self.log('test_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)

        output = [predictions,Y]

        return output

    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['V06','V12']

        cf_matrix = confusion_matrix(y_true, y_pred)

        fig = px.imshow(cf_matrix,labels=dict(x="Predicted condition", y="Actual condition"),x=target_names,y=target_names)
        fig.update_xaxes(side="top")
        fig.show()

        print(y_pred)
        print(y_true)
        print(classification_report(y_true, y_pred, target_names=target_names))

    def GetView(self,meshes,index):

        phong_renderer = self.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)

        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face   
        pix_to_face = pix_to_face.permute(0,3,1,2)
        return pix_to_face
