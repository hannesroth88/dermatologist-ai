import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import Config as config


class DataLoaders(object):
    def createDataloaders(self, config):
        print('')
        print('####################')
        print('Create Dataloaders')
        print('####################')

        # see https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # load and transform data using ImageFolder
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.08, 1), ratio=(1, 1)),
                #         transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'valid': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),  # because of size missmatch
                transforms.ToTensor(),
                normalize
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),  # because of size missmatch
                transforms.ToTensor(),
                normalize
            ]),
        }
        # image_datasets = {x: datasets.ImageFolder(config.dirs[x], transform=data_transforms[x])
        #                   for x in ['train', 'valid', 'test']}
        image_datasets = {x: ImageFolderWithPaths(root=config.dirs[x], transform=data_transforms[x])
                          for x in ['train', 'valid', 'test']}
        # prepare data loaders
        loaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=True)
            for x in ['train', 'valid', 'test']}
        dataset_sizes = {x: len(image_datasets[x])
                         for x in ['train', 'valid', 'test']}
        loaders['test1by1'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=1,
                                                          num_workers=config.num_workers,
                                                          shuffle=True)
        class_names = image_datasets['train'].classes

        # print out some data stats
        print('Num of Images: ' + str(dataset_sizes))
        print('Num of classes: ', len(class_names), ' -> ', class_names)

        return loaders, dataset_sizes, class_names


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
