import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import pandas as pd
from boson_sampler import BosonSampler
from utils import accuracy
import perceval as pcvl
import perceval.providers.scaleway as scw  # Uncomment to allow running on scaleway

class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self,bs=None,bs_size=None,device = 'cpu'):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.pre_net = nn.Linear(512, bs_size)
        self.bs = bs
        self.device = device

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        # q_out = torch.Tensor(0, 10)
        # q_out = q_out.to(self.device)
        q_out = []
        for elem in pre_out:
            embs = self.bs.embed(elem,1000).unsqueeze(0)
            x = nn.Linear(embs.shape[-1],10)(embs) 
            q_out.append(x)
        q_out = torch.cat([q_out[i] for i in range(10)],dim=0)
        # return nn.Linear(q_out.shape[-1],10)(q_out)
        # print(q_out.shape)
        return q_out
    
class MnistModel(nn.Module):
    def __init__(self, device = 'cpu', embedding_size = 0):
        super().__init__()
        input_size = 28 * 28
        num_classes = 10
        self.device = device
        self.embedding_size = embedding_size
        if self.embedding_size:
            input_size += embedding_size #considering 30 photons and 2 modes
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, xb, emb = None):
        xb = xb.reshape(-1, 784)
        if self.embedding_size and emb is not None:
            # concatenation of the embeddings and the input images
            xb = torch.cat((xb,emb),dim=1)
        out = self.linear(xb)
        return(out)
    
    def training_step(self, batch, emb = None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device)) ## Generate predictions
        else:
            out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels) ## Calculate the loss
        acc = accuracy(out, labels)
        return loss, acc
    
    def validation_step(self, batch, emb =None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device)) ## Generate predictions
        else:
            out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return({'val_loss':loss, 'val_acc': acc})
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        return result['val_loss'], result['val_acc']

# evaluation of the model
# evaluation of the model
def evaluate(model, val_loader, bs: BosonSampler = None):
    # if model.embedding_size:
    #     outputs = []
    #     for step, batch in enumerate(tqdm(val_loader)):
    #         # embedding in the BS
    #         images, labs = batch
    #         images = images.squeeze(0).squeeze(0)
    #         t_s = time.time()
    #         embs = bs.embed(images,1000)
    #         outputs.append(model.validation_step(batch, emb=embs.unsqueeze(0)))
    # else:
    outputs = [model.validation_step(batch) for batch in val_loader]
    return(model.validation_epoch_end(outputs))