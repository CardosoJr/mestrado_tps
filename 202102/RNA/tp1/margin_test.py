import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from affinity_loss import affinity_loss
from margin_loss_pt import LargeMarginLoss
from afinity_loss_pt import *
import datasets_pt as ds
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import matplotlib as plt

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

## mnist n_classe = 10 sigma = 10
## cifar10 n_classe = 10 sigma = 90
## cifar100 n_classe = 100 sigma = 90

##############################################################################################
########## DEFINE AFFINITY STRUCTURE



##############################################################################################
########## DEFINE MAX MARGIN STRUCTURE



##############################################################################################
########## TRAIN - TEST LOOPS

def get_data(problem, batch_size, imbal_class_prop):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    if problem == 'CIFAR10':
        train_dataset_imbalanced = ds.ImbalancedCIFAR10(
                imbal_class_prop, root='.', train=True, download=True, transform=transform)
        test_dataset_imbalanced = ds.ImbalancedCIFAR10(
            imbal_class_prop,
            root='.',
            train=False,
            download=True,
            transform=transform)

        _, train_class_counts = train_dataset_imbalanced.get_labels_and_class_counts()
        _, test_class_counts = test_dataset_imbalanced.get_labels_and_class_counts()
        
        train_loader_imbalanced = DataLoader(
            dataset=train_dataset_imbalanced,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available())

        test_loader_imbalanced = DataLoader(
            dataset=test_dataset_imbalanced,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available())
    else:
        pass

    return train_loader_imbalanced, test_loader_imbalanced, train_class_counts, test_class_counts

def train(epoch, model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Push data and target to GPU, if available
        data = data.to(device)
        target = target.to(device)

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Weight update
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:  # print every 100 mini-batches
            print('[{}, {}] loss: {:.3f}'.format(epoch + 1, batch_idx + 1,
                                                 running_loss / 100))
            running_loss = 0.0

def train_large_margin(epoch, model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Push data and target to GPU, if available
        data = data.to(device)
        one_hot = torch.zeros(len(target), 10).scatter_(1, target.unsqueeze(1), 1.).float()
        one_hot = one_hot.cuda()        
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        output, feature_maps = model(data)
        loss = criterion(output, one_hot, feature_maps)
        # Backward pass
        loss.backward()
        # Weight update
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:  # print every 100 mini-batches
            print('[{}, {}] loss: {:.3f}'.format(epoch + 1, batch_idx + 1,
                                                 running_loss / 100))
            running_loss = 0.0

def evaluate(model, model_name, test_loader, nb_classes, class_names, plot_confusion_mat=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Push data and target to GPU, if available
            data = data.to(device)
            target = target.to(device)
            if model_name == 'large_margin':
                output, *_ = model(data)
            else:
                output = model(data)
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    # Calculate global accuracy
    accuracy = 100 * correct / total
    # Print statistics
    print('Accuracy of the model {:.3f}%'.format(accuracy))

def train_model(model_name, problem, epochs, batch_size, imbal_class_prop):
    train_loader_imbalanced, test_loader_imbalanced, train_class_counts, test_class_counts = get_data(problem, batch_size, imbal_class_prop)
    if model_name == 'large_margin':
        criterion =  LargeMarginLoss(
                        gamma=10000,
                        alpha_factor=4,
                        top_k=1,
                        dist_norm=np.inf)
        model = NetLargeMargin()
        train_method = train_large_margin
    else:
        criterion = affinity_loss(0.75)
        if problem == "CIFAR10":
            model = NetAffinityCifar()
        else:
            model = NetAffinityMnist()
        train_method = train

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(epochs):
        train_method(epoch, model, train_loader_imbalanced, optimizer, criterion)
        evaluate(model, model_name, test_loader_imbalanced, plot_confusion_mat  = False)

class NetLargeMargin(nn.Module):
    def __init__(self):
        super(NetLargeMargin, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True)
        )
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flatten = conv2.view(x.shape[0], -1)        
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        return fc2, [conv1, conv2]

class NetAffinityCifar(nn.Module):
    def __init__(self):
        super(NetAffinityCifar, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True)
        )
        self.fc2 = nn.Linear(512, 10)
        self.last = ClusteringAffinity(10, 1, 90)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flatten = conv2.view(x.shape[0], -1)        
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        output = self.last(fc2) 
        return output

class NetAffinityMnist(nn.Module):
    def __init__(self):
        super(NetAffinityMnist, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True)
        )
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flatten = conv2.view(x.shape[0], -1)        
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        output = self.last(fc2) 
        self.last = ClusteringAffinity(10, 1, 10)
        return output


