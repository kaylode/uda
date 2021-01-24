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


class Supervised_Trainer():
    def __init__(self, cfg):
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.classes = cfg.classes
        self.model = EfficientNet.from_pretrained(cfg.model_name, num_classes = len(self.classes)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1,)
        self.num_epochs = cfg.num_epochs
        self.trainloader, self.valloader = dataset.cifar10_supervised_dataloaders(cfg)
        self.print_per_iter = cfg.print_per_iter
        self.checkpoint_path = cfg.checkpoint_path
        self.iters = 0
        self.epoch = 0
        self.num_iters = (self.num_epochs+1) * len(self.trainloader)

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
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            end_time = time.time()

            running_time += (end_time - start_time)
            running_loss['SUP'] += loss.item()
            running_loss['T'] += loss.item()
            self.iters = len(self.trainloader)*self.epoch + idx + 1
            
            if self.iters % self.print_per_iter == 0:
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

            

            
    def save_model(self, name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        torch.save(self.model.state_dict(), name)

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
                preds = torch.argmax(outputs, dim=1)
                corrects += (preds == targets).sum()
                sample_size += outputs.size(0)
        
        acc = corrects.cpu()*1.0/sample_size
        total_loss /= sample_size

        acc = np.round(acc, 5)
        total_loss = np.round(total_loss, 5)
        print(f"Validation: Loss: {total_loss} || Accuracy: {acc}")
        self.save_model(os.path.join(self.checkpoint_path, f'supervised_{self.epoch}_{acc}.pth'))

    def fit(self):
        for self.epoch in range(self.num_epochs):
            self.train_epoch()
            self.val_epoch()
    

if __name__ == '__main__':
    config = Config('configs/supervised.yaml')
    trainer = Supervised_Trainer(config)
    trainer.fit()