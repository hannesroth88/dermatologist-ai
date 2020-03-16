from collections import OrderedDict

import torch
import torchvision
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

        # model = models.vgg16(pretrained=True)
        model = models.resnet18(pretrained=True)
        # Freeze training for all "feature" layers
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        classifier = nn.Sequential(OrderedDict([('drop', nn.Dropout(0.5)),
                                                ('fc1', nn.Linear(n_inputs, 256)),
                                                ('relu', nn.ReLU()),
                                                #('drop', nn.Dropout(0.5)),
                                                ('fc2', nn.Linear(256, len(class_names)))]))
        model.fc = classifier
        # move model to GPU if CUDA is available
        if config.use_cuda:
            print("CUDA: moving model to GPU")
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
