import torch


class Config(object):
    batch_size = 20
    num_workers = 0
    epochs = 100
    learning_rate = 0.0005

    # define training and test data directories
    dirs = {'train': './data/train/',
            'valid': './data/valid/',
            'test': './data/test/'}

    # check if CUDA is available
    use_cuda = torch.cuda.is_available()
    # use_cuda = False


