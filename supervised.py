import os

import time
import argparse

import torch
from torch import nn
from torch.backends import cudnn

from torch.utils.tensorboard import SummaryWriter

import dataset as dataset
from efficientnet_pytorch import EfficientNet


class Supervised_Trainer():
    def __init__(self, cfg):
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1,)
        self.epoch = cfg.epoch
        self.trainloader, self.valloader = dataset.cifar10_supervised_dataloaders()
        self.print_per_iter = 10
        self.iters = 0
        self.num_iters = (num_epochs+1) * len(self.sup_trainloader)

    def train_epoch(self):
        self.model.train()
        running_time = 0
        running_loss = {
            "SUP": 0,
            "T": 0
        }

        for idx, batch in enumerate(self.trainloader):
            start_time = time.time()
            self.optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.squeeze(1).to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            end_time = time.time()

            running_time += (end_time - start_time)
            running_loss['SUP'] += loss.item()
            running_loss['T'] += loss.item()
            self.iters = len(self.trainloader)*self.epoch + idx + 1
            
            if idx % self.print_per_iter == 0:
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(self.epoch, self.num_epochs, self.iters, self.num_iters, loss_string, running_time))
                
                running_time = 0
                running_loss = {
                    "SUP": 0,
                    "T": 0
                }

    def val_epoch(self):
        self.model.eval()
        corrects = 0
        sample_size = 0
        total_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(self.valloader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                preds = torch.argmax(outputs)
                corrects += (pred == target).sum()
                sample_size += outputs.size(0)
        
        acc = corrects*1.0/sample_size
        total_loss /= sample_size
        print(f"Validation: Loss: {total_loss} || Accuracy: {acc}")


    def fit(self):
        for epoch in range(self.epoch):
            self.train_epoch()
            self.val_epoch()
    