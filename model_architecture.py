import torch
import torchvision.models as models
import torch.nn as nn
from config import Config as config
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np



class ModelArchitecture(object):

    def createModelResnet18(self, config, class_names):

        print('')
        print('####################')
        print('Create Model')
        print('####################')

        # define VGG16 model
        # model_transfer = models.vgg16(pretrained=True)
        model = models.resnet18(pretrained=True)
        # Freeze training for all "feature" layers
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Linear(n_inputs, len(class_names))
        # move model to GPU if CUDA is available
        if config.use_cuda:
            model = model.cuda()

        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of Parameters: ", str(count))

        return model

    def setOptimizer(self, model, learning_rate):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

        # Define a learningRate scheduler
        scheduler_transfer = ReduceLROnPlateau(optimizer, 'min', verbose=True)
        return optimizer, scheduler_transfer, criterion

