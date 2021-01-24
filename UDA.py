import os
import time
import argparse
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from configs import Config
import dataset as dataset
from efficientnet_pytorch import EfficientNet


class Unsupervised_Trainer():
    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10).to(self.device)
        self.sup_criterion = nn.CrossEntropyLoss().to(self.device)
        self.unsup_criterion = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.num_epochs = cfg.num_epochs
        self.sup_trainloader, self.unsup_trainloader, self.unsup_aug_trainloader, self.valloader = dataset.cifar10_unsupervised_dataloaders(cfg)

        self.sup_iter = iter(self.sup_trainloader)
        self.unsup_iter = iter(self.unsup_trainloader)
    
        self.print_per_iter = 10
        self.iters = 0
        self.epoch = 0
        self.lamb = cfg.lamb
        self.num_iters = (self.num_epochs+1) * len(self.sup_trainloader)

    def train_epoch(self):
        self.model.train()
        running_time = 0
        running_loss = {
            "SUP": 0,
            "UNSUP":0,
            "T": 0
        }

        for idx, sup_batch in enumerate(self.unsup_aug_trainloader):
            start_time = time.time()
            self.optimizer.zero_grad()
            unsup_aug_inputs, _ = sup_batch

            # Unsup batch
            try:
                sup_inputs, sup_targets = next(self.sup_iter)
                unsup_inputs, _ = next(self.unsup_iter)
            except StopIteration:
                self.sup_iter = iter(self.sup_trainloader)
                self.unsup_iter = iter(self.unsup_trainloader)
                sup_inputs, sup_targets = next(self.sup_iter)
                unsup_inputs, _ = next(self.unsup_iter)
                
            sup_inputs = sup_inputs.to(self.device)
            sup_targets = sup_targets.to(self.device)
            unsup_inputs = unsup_inputs.to(self.device)
            unsup_aug_inputs = unsup_aug_inputs.to(self.device)

            sup_outputs = self.model(sup_inputs)

            unsup_y_pred = self.model(unsup_inputs).detach()
            unsup_y_probas = torch.softmax(unsup_y_pred, dim=-1)

            unsup_aug_y_pred = self.model(unsup_aug_inputs)
            unsup_aug_y_probas = torch.log_softmax(unsup_aug_y_pred, dim=-1)

            unsup_loss = self.unsup_criterion(unsup_aug_y_probas, unsup_y_probas)
            sup_loss = self.sup_criterion(sup_outputs, sup_targets)
            loss = sup_loss + self.lamb * unsup_loss

            loss.backward()
            self.optimizer.step()
            
            end_time = time.time()

            running_time += (end_time - start_time)
            running_loss['SUP'] += sup_loss.item()
            running_loss['UNSUP'] += unsup_loss.item()
            running_loss['T'] += loss.item()
            self.iters = len(self.sup_trainloader)*self.epoch + idx + 1
            
            if idx % self.print_per_iter == 0:
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(self.epoch, self.num_epochs, self.iters, self.num_iters, loss_string, running_time))
                
                running_time = 0
                running_loss = {
                    "SUP": 0,
                    "UNSUP":0,
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
                loss = self.sup_criterion(outputs, targets)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                corrects += (preds == targets).sum()
                sample_size += outputs.size(0)
        
        acc = corrects.cpu()*1.0/sample_size
        total_loss /= sample_size

        acc = np.round(acc, 5)
        total_loss = np.round(total_loss, 5)
        print(f"Validation: Loss: {total_loss} || Accuracy: {acc}")


    def fit(self):
        for self.epoch in range(self.num_epochs):
            self.train_epoch()
            self.val_epoch()
    

if __name__ == '__main__':
    config = Config('configs/unsupervised.yaml')
    trainer = Unsupervised_Trainer(config)
    trainer.fit()