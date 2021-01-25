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
from loggers import Logger

class Unsupervised_Trainer():
    def __init__(self, cfg):
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.classes = cfg.classes
        self.model = EfficientNet.from_pretrained(cfg.model_name, num_classes=len(self.classes)).to(self.device)
        self.sup_criterion = nn.CrossEntropyLoss().to(self.device)
        self.unsup_criterion = nn.KLDivLoss(reduction='none').to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.num_epochs = cfg.num_epochs
        self.sup_trainloader, self.unsup_trainloader, self.unsup_aug_trainloader, self.valloader = dataset.cifar10_unsupervised_dataloaders(cfg)
        self.sup_batch_size = cfg.sup_batch_size
        self.unsup_batch_size = cfg.unsup_batch_size
        self.sup_iter = iter(self.sup_trainloader)
        self.unsup_iter = iter(self.unsup_trainloader)
        self.checkpoint_path = cfg.checkpoint_path
        
        self.print_per_iter = cfg.print_per_iter
        self.iters = 0
        self.epoch = 0
        self.lamb = cfg.lamb
        self.temperature = cfg.temperature if cfg.temperature > 0 else 1.
        self.beta = cfg.beta
        self.num_iters = (self.num_epochs+1) * len(self.sup_trainloader)
        self.logger = Logger(log_dir=cfg.log_dir)

    def train_epoch(self):
        self.model.train()
        running_time = 0
        running_loss = {
            "SUP": 0,
            "UNSUP":0,
            "T": 0
        }

        for idx, unsup_aug_batch in enumerate(self.unsup_aug_trainloader):
            start_time = time.time()
            self.optimizer.zero_grad()
            unsup_aug_inputs, _ = unsup_aug_batch

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

            # confidence-based masking
            if self.beta != 0:
                unsup_loss_mask = torch.max(unsup_y_probas, dim=-1)[0] > self.beta
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
            else:
                unsup_loss_mask = torch.ones(unsup_inputs.shape[0], dtype=torch.float32)
            unsup_loss_mask = unsup_loss_mask.to(self.device)
            
            # temperature scaling
            unsup_aug_y_probas = torch.log_softmax(unsup_aug_y_pred/self.temperature, dim=-1)

            unsup_loss = self.unsup_criterion(unsup_aug_y_probas, unsup_y_probas)
            unsup_loss = torch.sum(unsup_loss, dim=-1)

            sup_loss = self.sup_criterion(sup_outputs, sup_targets)
            
            num_unsup = torch.sum(unsup_loss_mask, dim=-1)
            if num_unsup == 0.:
                unsup_loss = torch.FloatTensor([0.])
                loss = sup_loss            
            else:
                unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.sum(unsup_loss_mask, dim=-1)
                loss = sup_loss + self.lamb * unsup_loss

            loss.backward()
            self.optimizer.step()
            
            end_time = time.time()

            running_time += (end_time - start_time)
            running_loss['SUP'] += sup_loss.item()
            running_loss['UNSUP'] += unsup_loss.item()
            running_loss['T'] += loss.item()
            self.iters = len(self.sup_trainloader)*self.epoch + idx + 1
            
            if self.iters % self.print_per_iter == 0:
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(self.epoch, self.num_epochs, self.iters, self.num_iters, loss_string, running_time))
                self.logging({"Training Loss" : running_loss['T']/ self.print_per_iter,})
                running_time = 0
                running_loss = {
                    "SUP": 0,
                    "UNSUP":0,
                    "T": 0
                }

    def logging(self, logs):
        tags = [l for l in logs.keys()]
        values = [l for l in logs.values()]
        self.logger.write(tags= tags, values= values)

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
                loss = self.sup_criterion(outputs, targets)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                corrects += (preds == targets).sum()
                sample_size += outputs.size(0)
        
        acc = corrects.cpu()*1.0/sample_size
        acc = np.round(float(acc), 5)
        total_loss = np.round(total_loss, 5) /sample_size
        print(f"Validation: Loss: {total_loss} || Accuracy: {acc}")
        self.logging({
            "Validation Loss" : total_loss/ len(self.valloader),
            "Validation Accuracy": acc})
        self.save_model(os.path.join(self.checkpoint_path, f'supervised_{self.epoch}_{acc}.pth'))


    def fit(self):
        print(f"Training semi-supervisedly with {len(self.sup_trainloader)*self.sup_batch_size} labelled and {len(self.unsup_trainloader)*self.unsup_batch_size} unlabelled.")
        for self.epoch in range(self.num_epochs):
            self.train_epoch()
            self.val_epoch()
    

if __name__ == '__main__':
    config = Config('configs/unsupervised.yaml')
    trainer = Unsupervised_Trainer(config)
    trainer.fit()