# for the Boson Sampler
import perceval as pcvl
#import perceval.providers.scaleway as scw  # Uncomment to allow running on scaleway

# for the machine learning model
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from boson_sampler import BosonSampler
from utils import MNIST_partial, accuracy, plot_training_metrics
from model import *
import csv

session = None
# to run a remote session on Scaleway, uncomment the following and fill project_id and token
# session = scw.Session(
#                    platform="sim:sampling:p100",  # or sim:sampling:h100
#                    project_id=""  # Your project id,
#                    token=""  # Your personal API key
#                    )

# start session
if session is not None:
    session.start()
# definition of the BosonSampler
# here, we use 30 photons and 2 modes

bs = BosonSampler(24,2, postselect = 2, session = session)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# dataset from csv file, to use for the challenge
train_dataset = MNIST_partial(split = 'train')
val_dataset = MNIST_partial(split='val')

# definition of the dataloader, to process the data in the model
# here, we need a batch size of 1 to use the boson sampler
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size, shuffle = False)

def training_step(model, batch, emb = None):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    images.requires_grad = True  # Only needed for special cases
    # if self.embedding_size:
    #     out = self(images, emb.to(self.device)) ## Generate predictions
    # else:
    # images = images.reshape(-1,1,28,28)
    # print(images.shape)

    # out = model(images) ## Generate predictions
    # loss = F.cross_entropy(out, labels)
    # acc = accuracy(out, labels, task="multiclass", num_classes=10)
    loss, acc = model((images, labels)) ## Generate predictions
    
    # loss = F.cross_entropy(out, labels) ## Calculate the loss
    # acc = accuracy(out, labels)
    return loss, acc

def validation_step(model, batch, emb =None):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    images.requires_grad = True  # Only needed for special cases
    # if self.embedding_size:
    #     out = self(images, emb.to(self.device)) ## Generate predictions
    # # else:
    # out = model(images) ## Generate predictions
    # loss = F.cross_entropy(out, labels)
    # acc = accuracy(out, labels, task="multiclass", num_classes=10)
    loss, acc = model((images,labels)) ## Generate predictions
    return({'val_loss':loss, 'val_acc': acc})

def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})

def epoch_end(epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    return result['val_loss'], result['val_acc']

# training loop
def fit(epochs, lr, model, train_loader, val_loader, bs: BosonSampler, opt_func = torch.optim.SGD, filename='tmp_file.txt'):
    history = []
    optimizer = opt_func([{'params': model.model.fc.parameters(), 'lr': lr},{'params': model.model.conv1.parameters(), 'lr': lr}])
    # creation of empty lists to store the training metrics
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for epoch in range(epochs):
        training_losses, training_accs = 0, 0
        ## Training Phase
        for step, batch in enumerate(tqdm(train_loader)):
            # # embedding in the BS
            # if model.embedding_size:
            #     images, labs = batch
            #     images = images.squeeze(0).squeeze(0)
            #     t_s = time.time()
            #     embs = bs.embed(images,1000)
            #     loss,acc = model.training_step(batch,emb = embs.unsqueeze(0))

            # else:
            
            loss,acc = training_step(model, batch)
            # loss.requires_grad = True
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Backpropagation with boson sampling noise
            # for param in model.parameters():
            #     param.grad = param.grad + bs(param.grad.shape, scale=0.001)

            training_losses+=int(loss.detach())
            training_accs+=int(acc.detach())
            # if model.embedding_size and step%100==0:
            #     print(f"STEP {step}, Training-acc = {training_accs/(step+1)}, Training-losses = {training_losses/(step+1)}")
        
        ## Validation phase
        outputs = [validation_step(model, batch) for batch in val_loader]
        result = (validation_epoch_end(outputs))
        # result = evaluate(model, val_loader, bs)
        validation_loss, validation_acc = result['val_loss'], result['val_acc']
        epoch_end(epoch, result)
        history.append(result)

        ## summing up all the training and validation metrics
        training_loss = training_losses/len(train_loader)
        training_accs = training_accs/len(train_loader)
        train_loss.append(training_loss)
        train_acc.append(training_accs)
        val_loss.append(validation_loss)
        val_acc.append(validation_acc)

        # plot training curves
        

    with open(filename, 'w') as f:
        csv.writer(f, delimiter=' ').writerows([train_acc,val_acc,train_loss,val_loss])
        # plot_training_metrics(train_acc,val_acc,train_loss,val_loss)
    return(history)
from pre_trained import *

model = CIFAR10Module(minst=True, quantum=bs)
model.model.fc = DressedQuantumNet(bs, bs.nb_parameters, dropout=False, pos=False)

for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Unfreeze the last layer
for param in model.model.fc.parameters():
    param.requires_grad = True

for param in model.model.conv1.parameters():
    param.requires_grad = True

pretrained_dict = torch.load("state_dicts/resnet18.pt")  # Path to saved weights
model_dict = model.state_dict()

# Remove 'conv1' weights from pretrained_dict to avoid shape mismatch
pretrained_dict = {k: v for k, v in pretrained_dict.items() if "conv1" or "fc" not in k}

# Update the current model dictionary
model_dict.update(pretrained_dict)

# Load the modified weights
model.load_state_dict(model_dict, strict=False) 
model.to(device)

# train the model with the chosen parameters
experiment = fit(epochs = 20, lr = 0.001, model = model, train_loader = train_loader, val_loader = val_loader, bs=bs, filename='tmp_file.txt')
# model = CIFAR10Module(minst=True, quantum=bs)
# model.model.fc = DressedQuantumNet(bs, bs.nb_parameters, dropout=True, pos=False)

# for param in model.parameters():
#     param.requires_grad = False  # Freeze all layers

# # Unfreeze the last layer
# for param in model.model.fc.parameters():
#     param.requires_grad = True

# for param in model.model.conv1.parameters():
#     param.requires_grad = True

# pretrained_dict = torch.load("state_dicts/resnet18.pt")  # Path to saved weights
# model_dict = model.state_dict()

# # Remove 'conv1' weights from pretrained_dict to avoid shape mismatch
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if "conv1" or "fc" not in k}

# # Update the current model dictionary
# model_dict.update(pretrained_dict)

# # Load the modified weights
# model.load_state_dict(model_dict, strict=False) 
# model.to(device)

# # train the model with the chosen parameters
# experiment = fit(epochs = 20, lr = 0.001, model = model, train_loader = train_loader, val_loader = val_loader, bs=bs, filename='tmp_file1.txt')

model = CIFAR10Module(minst=True, quantum=bs)
model.model.fc = DressedQuantumNet(bs, bs.nb_parameters, dropout=False, pos=True)

for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Unfreeze the last layer
for param in model.model.fc.parameters():
    param.requires_grad = True

for param in model.model.conv1.parameters():
    param.requires_grad = True

pretrained_dict = torch.load("state_dicts/resnet18.pt")  # Path to saved weights
model_dict = model.state_dict()

# Remove 'conv1' weights from pretrained_dict to avoid shape mismatch
pretrained_dict = {k: v for k, v in pretrained_dict.items() if "conv1" or "fc" not in k}

# Update the current model dictionary
model_dict.update(pretrained_dict)

# Load the modified weights
model.load_state_dict(model_dict, strict=False) 
model.to(device)

# train the model with the chosen parameters
experiment = fit(epochs = 20, lr = 0.001, model = model, train_loader = train_loader, val_loader = val_loader, bs=bs, filename='tmp_file2.txt')

# model = CIFAR10Module(minst=True, quantum=bs)
# model.model.fc = DressedQuantumNet(bs, bs.nb_parameters, dropout=True, pos=True)

# for param in model.parameters():
#     param.requires_grad = False  # Freeze all layers

# # Unfreeze the last layer
# for param in model.model.fc.parameters():
#     param.requires_grad = True

# for param in model.model.conv1.parameters():
#     param.requires_grad = True

# pretrained_dict = torch.load("state_dicts/resnet18.pt")  # Path to saved weights
# model_dict = model.state_dict()

# # Remove 'conv1' weights from pretrained_dict to avoid shape mismatch
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if "conv1" or "fc" not in k}

# # Update the current model dictionary
# model_dict.update(pretrained_dict)

# # Load the modified weights
# model.load_state_dict(model_dict, strict=False) 
# model.to(device)

# # train the model with the chosen parameters
# experiment = fit(epochs = 20, lr = 0.001, model = model, train_loader = train_loader, val_loader = val_loader, bs=bs, filename='tmp_file3.txt')