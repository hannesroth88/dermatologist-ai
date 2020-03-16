import datetime

# Set PIL to be tolerant of image files that are truncated.
import torch
from PIL import ImageFile

from config import Config as config
from dataloaders import DataLoaders
from model_architecture import ModelArchitecture
from pipeline import Pipeline
import pprint
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Application(object):

    # init method or constructor
    def __init__(self):
        print('####################')
        print('INITIALIZE')
        print('####################')
        self.config = config()
        pp = pprint.PrettyPrinter(indent=0)
        pp.pprint(config.__dict__)
        # Initialize all
        self.loaders = None
        self.dataset_sizes = None
        self.class_names = None
        self.model = None
        self.optimizer = None
        self.scheduler_transfer = None
        self.criterion = None
        self.losses = None

        print('####################')
        print('START')
        print('####################')

        # Start Application
        Application.start(self)

        print('####################')
        print('END')
        print('####################')

    def start(self):
        torch.manual_seed(0)
        np.random.seed(0)

        # Create dataloaders and update to self
        dataloaders = DataLoaders()
        self.loaders, self.dataset_sizes, self.class_names = dataloaders.createDataloaders(self.config)

        # Create Model
        modelArchitecture = ModelArchitecture()
        self.model = modelArchitecture.createModelResnet18(self.config, self.class_names)
        self.optimizer, self.scheduler, self.criterion = modelArchitecture.setOptimizer(self.model,
                                                                                        self.config.learning_rate)
        # print(self.model)

        pipeline = Pipeline()
        # Train Model
        self.model = pipeline.train(self.config, self.loaders, self.model, self.optimizer, self.criterion, lr_scheduler=self.scheduler)

        # Plot Losses
        pipeline.plotLosses('./results/' + config.model_name + '/losses_' + config.model_index + '.pt')

        # Test Model
        # load the model that got the best validation accuracy
        self.model.load_state_dict(torch.load('./results/' + config.model_name + '/model_' + config.model_index + '.pt'))
        pipeline.testModelAndWriteCsv(self.config, self.model, self.loaders, './results/' + config.model_name + '/test_' + config.model_index + '.csv')
        pipeline.testModelAccuracy(self.config, self.loaders, self.model, self.criterion)

        # Get Scores
        pipeline.getScores('./results/' + config.model_name + '/test_' + config.model_index + '.csv')


App = Application()
