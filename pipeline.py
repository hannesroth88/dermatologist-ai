import csv
import datetime

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torch.nn as nn
from config import Config as config
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import matplotlib.pyplot as plt


class Pipeline(object):

    def train(self, config, loaders, model, optimizer, criterion, save_path, lr_scheduler=None):
        print('')
        print('####################')
        print('Train and Validate Model')
        print('####################')
        print('Start:', str(datetime.datetime.now()))

        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf
        train_losses, valid_losses = [], []

        # Time tracking start
        time_start = time.time()

        for epoch in range(1, config.epochs + 1):
            print("Start epoch:", epoch)
            # do iteration with lots of work here
            # initialize variables to monitor training and validation loss
            # do long-running work here

            train_loss = 0.0
            valid_loss = 0.0

            time_start_epoch = time.time()

            ###################
            # train the model #
            ###################
            model.train()
            for batch_idx, (data, target) in enumerate(loaders['train']):
                # move to GPU
                if config.use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## find the loss and update the model parameters accordingly
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # record the average training loss
                train_loss += loss.item() * data.size(0)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
            # calculate average losses
            train_loss = train_loss / len(loaders['train'].dataset)
            # save losses
            train_losses.append(train_loss)
            ######################

            ######################
            # validate the model #
            ######################
            model.eval()
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                if config.use_cuda:
                    data, target = data.cuda(), target.cuda()
                # update the average validation loss
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)
            # calculate average losses
            valid_loss = valid_loss / len(loaders['valid'].dataset)
            # save losses
            valid_losses.append(valid_loss)
            ######################

            # Adjust Learning rate if scheduler is set
            if lr_scheduler is not None:
                lr_scheduler.step(valid_loss)

            # print training/validation statistics
            print('Epoch: {} took {:.2f} seconds. \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                time.time() - time_start_epoch,
                train_loss,
                valid_loss
            ))

            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), save_path)
                valid_loss_min = valid_loss

            losses = {'train_losses': train_losses, 'valid_losses': valid_losses}
            torch.save(losses, './results/losses.pt')

        # Show result
        print(
            f"num epochs:{config.epochs} it took {(time.time() - time_start):.0f} seconds. Minimum validation loss: {valid_loss_min:.3f}")

        print('End:', str(datetime.datetime.now()))
        # return trained model
        return model

    def plotLosses(self, plotPath):

        print('')
        print('####################')
        print('Plot Losses')
        print('####################')

        # Compare Train and Validation Loss
        # Plot losses model_transfer

        plot1 = torch.load(plotPath)

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        # Print Losses
        ax.plot(np.arange(1, len(plot1['train_losses']) + 1, 1), plot1['train_losses'], label='train')
        ax.plot(np.arange(1, len(plot1['valid_losses']) + 1, 1), plot1['valid_losses'], label='valid')
        # ax.set_xticks(epoch)
        ax.legend()
        # set the xlim
        ax.set_xlim(2, 100)
        # ax.set_ylim(0,5)
        plt.title("Losses - Train vs. Valid")
        plt.show()

    def testModelAccuracy(self, config, loaders, model, criterion):
        print('')
        print('####################')
        print('Test Accuracy')
        print('####################')
        # monitor test loss and accuracy
        test_loss = 0.
        correct = 0.
        total = 0.

        model.eval()
        for batch_idx, (data, target, path) in enumerate(loaders['test']):
            # move to GPU
            if config.use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

        print('Test Loss: {:.6f}'.format(test_loss))
        print('Test Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))

    def testModelAndWriteCsv(self, model, loaders, path_csv):

        print('')
        print('####################')
        print('Write to CSV')
        print('####################')

        f = open(path_csv, 'w', newline='')

        with f:
            header = ['Id', 'task_1', 'task_2', 'label']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for batch_idx, (data, target, path) in enumerate(loaders['test1by1']):
                model.eval()
                # move to GPU
                if config.use_cuda:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # convert output probabilities to predicted class
                # pred = output.data.max(1, keepdim=True)[1]
                # pred_melanoma = F.relu(output.data[0, 0])
                # pred_seborrheic_keratosis = F.relu(output.data[0, 2])
                pred_melanoma = output.data[0, 0]
                pred_seborrheic_keratosis = output.data[0, 2]
                # print('Id:', path, '    task_1:', str(pred_melanoma.cpu().numpy()), '    task_2:', str(pred_seborrheic_keratosis.cpu().numpy()), '    label', str(target.data.cpu().numpy()[0]))
                # target = target.cpu().numpy()
                writer.writerow(
                    {'Id': path, 'task_1': str(pred_melanoma.cpu().numpy()),
                     'task_2': str(pred_seborrheic_keratosis.cpu().numpy()),
                     'label': str(target.data.cpu().numpy()[0])})

                if batch_idx % 10 == 0:
                    print("write next line (", batch_idx, "/600)")
