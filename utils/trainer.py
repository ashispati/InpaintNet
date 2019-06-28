import os
import time
import datetime
from tqdm import tqdm

from abc import ABC, abstractmethod
from torch import nn
import numpy as np

from tensorboard_logger import configure, log_value
import matplotlib.pyplot as plt

from utils.helpers import *


class Trainer(ABC):
    """
    Abstract base class which will serve as a NN trainer
    """
    def __init__(self, dataset,
                 model,
                 lr=1e-4,
                 early_stopping=False):
        """
        Initializes the trainer class
        :param dataset: torch Dataset object
        :param model: torch.nn object
        :param lr: float, learning rate
        """
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        self.early_stopping = False
        if early_stopping:
            self.early_stopping = True
            self.early_stopper = EarlyStopping()

    def train_model(self, batch_size, num_epochs, plot=False, log=False):
        """
        Trains the model
        :param batch_size: int,
        :param num_epochs: int,
        :param plot: bool, creates a matplotlib plot if TRUE
        :param log: bool, logs epoch stats for viewing in tensorboard if TRUE
        :return: None
        """
        # set-up plot and log parameters
        if log:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y-%m-%d_%H:%M:%S'
            )
            # configure tensorboard_logger for logging loss data
            configure(os.path.join('runs/' + self.model.__repr__() + st))
        if plot:
            # initialize plot parameters
            plot_parameters = self.plot_init()

        # get dataloaders
        (generator_train,
         generator_val,
         _) = self.dataset.data_loaders(
            batch_size=batch_size,
            split=(0.70, 0.20)
        )
        print('Num Train Batches: ', len(generator_train))
        print('Num Valid Batches: ', len(generator_val))

        # train epochs
        for epoch_index in range(num_epochs):
            # print LR details
            self.update_scheduler(epoch_index)
            
            # run training loop on training data
            self.model.train()
            mean_loss_train, mean_accuracy_train = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True)

            # run evaluation loop on validation data
            self.model.eval()
            mean_loss_val, mean_accuracy_val = self.loss_and_acc_on_epoch(
                data_loader=generator_val,
                epoch_num=epoch_index,
                train=False)

            data_element = {
                'epoch_index': epoch_index,
                'num_epochs': num_epochs,
                'mean_loss_train': mean_loss_train,
                'mean_accuracy_train': mean_accuracy_train,
                'mean_loss_val': mean_loss_val,
                'mean_accuracy_val': mean_accuracy_val
            }
            # plot / log parameters
            if log:
                # log value in tensorboard for visualization
                log_value('train_loss', mean_loss_train, epoch_index)
                log_value('train_accu', mean_accuracy_train, epoch_index)
                log_value('valid_loss', mean_loss_val, epoch_index)
                log_value('valid_accu', mean_accuracy_val, epoch_index)
            if plot:
                self.plot_epoch_stats(
                    **plot_parameters,
                    **data_element,
                )

            # print epoch stats
            self.print_epoch_stats(**data_element)
            # overwrite model after every epoch
            self.model.save()
            # save checkpoints
            if epoch_index > 0 and epoch_index % 10 == 0:
                self.model.save_checkpoint(epoch_index)
            # stop training for early stopping
            if self.early_stopping:
                self.early_stopper(mean_loss_val, self.model)
                if self.early_stopper.early_stop:
                    print("Early Stopping")
                    return

    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for an epoch
        :param data_loader: torch dataloader object
        :param epoch_num: int, used to change training schedule
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        mean_loss = 0
        mean_accuracy = 0
        for sample_id, batch in tqdm(enumerate(data_loader)):
            # process batch data
            batch_data = self.process_batch_data(batch)

            # zero the gradients
            self.zero_grad()

            # compute loss for batch
            loss, accuracy = self.loss_and_acc_for_batch(
                batch_data, epoch_num, train=train
            )

            # compute backward and step if train
            if train:
                loss.backward()
                self.step()

            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            if accuracy is not None:
                mean_accuracy += to_numpy(accuracy)

        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def zero_grad(self):
        """
        Zero the grad of the relevant optimizers
        :return:
        """
        self.optimizer.zero_grad()

    def step(self):
        """
        Perform the step update for all optimizers
        :return:
        """
        self.optimizer.step()

    @abstractmethod
    def loss_and_acc_for_batch(self, batch, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        pass

    @abstractmethod
    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: torch Variable or tuple of torch Variable objects
        """
        pass

    @abstractmethod
    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        pass

    @staticmethod
    def plot_init():
        fig, axarr = plt.subplots(2, sharex=True)
        fig.show()

        fig_parameters = {
            'fig': fig,
            'axarr': axarr,
            'x': [],
            'y_loss_train': [],
            'y_acc_train': [],
            'y_loss_val': [],
            'y_acc_val': [],
        }
        return fig_parameters

    @staticmethod
    def print_epoch_stats(epoch_index,
                          num_epochs,
                          mean_loss_train,
                          mean_accuracy_train,
                          mean_loss_val,
                          mean_accuracy_val):
        """
        Prints the epoch statistics
        :param epoch_index: int,
        :param num_epochs: int,
        :param mean_loss_train: float,
        :param mean_accuracy_train:float,
        :param mean_loss_val: float,
        :param mean_accuracy_val: float
        :return: None
        """
        print(
            f'Train Epoch: {epoch_index + 1}/{num_epochs}')
        print(f'\tTrain Loss: {mean_loss_train}'
              f'\tTrain Accuracy: {mean_accuracy_train * 100} %'
              )
        print(
            f'\tValid Loss: {mean_loss_val}'
            f'\tValid Accuracy: {mean_accuracy_val* 100} %'
        )

    @staticmethod
    def plot_epoch_stats(x, y_loss_train, y_acc_train,
                    y_loss_val, y_acc_val,
                    axarr, fig,
                    epoch_index,
                    num_epochs,
                    mean_loss_train,
                    mean_accuracy_train,
                    mean_loss_val,
                    mean_accuracy_val):
        x.append(epoch_index)
        y_loss_train.append(mean_loss_train)
        y_acc_train.append(mean_accuracy_train * 100)
        y_loss_val.append(mean_loss_val)
        y_acc_val.append(mean_accuracy_val * 100)
        axarr[0].plot(x, y_loss_train, 'r-', x, y_loss_val, 'r--')
        axarr[1].plot(x, y_acc_train, 'r-', x, y_acc_val, 'r--')
        fig.canvas.draw()
        plt.pause(0.001)

    @staticmethod
    def mean_crossentropy_loss(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return: float, loss
        """
        criteria = nn.CrossEntropyLoss(reduction='elementwise_mean')
        batch_size, seq_len, num_notes = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        weights = weights.contiguous().view(-1, num_notes)
        targets = targets.contiguous().view(-1)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_accuracy(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return float, accuracy
        """
        _, _, num_notes = weights.size()
        weights = weights.contiguous().view(-1, num_notes)
        targets = targets.contiguous().view(-1)

        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)

    @staticmethod
    def mean_l1_loss_rnn(weights, targets):
        """
        Evaluates the mean l1 loss
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.L1Loss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_mse_loss_rnn(weights, targets):
        """
        Evaluates the mean mse loss
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.MSELoss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_crossentropy_loss_alt(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return: float, loss
        """
        criteria = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_accuracy_alt(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return float, accuracy
        """
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience
    Code Courtesy: https://github.com/Bjarten/early-stopping-pytorch/
    """
    def __init__(self, patience=5, verbose=False):
        """
        Initialization method
        :param patience: int, number of epochs to wait
        :param verbose: bool, prints a message for each validation loss improvement if True
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop=False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if score - self.best_score < 1e-5:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.val_loss_min = val_loss
                self.counter = 0
