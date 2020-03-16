import torch


class Config(object):
    model_name = 'Resnet18'
    model_index = '2'
    batch_size = 20
    num_workers = 0
    epochs = 100
    learning_rate = 0.001

    # define training and test data directories
    dirs = {'train': './data/train/',
            'valid': './data/valid/',
            'test': './data/test/'}

    # check if CUDA is available
    use_cuda = torch.cuda.is_available()
    # use_cuda = False

