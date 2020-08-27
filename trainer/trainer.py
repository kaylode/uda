import os
import torch.nn as nn
import torch
from tqdm import tqdm
from .checkpoint import Checkpoint
import numpy as np
from loggers.loggers import Logger
from utils.utils import clip_gradient
import time


class Trainer(nn.Module):
    def __init__(self, 
                model, 
                trainloader, 
                valloader,
                **kwargs):

        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.trainloader = trainloader
        self.valloader = valloader
        self.metrics = model.metrics #list of metrics
        self.set_attribute(kwargs)
        
        

    def fit(self, num_epochs = 10 ,print_per_iter = None):
        self.num_epochs = num_epochs
        self.num_iters = (num_epochs+1) * len(self.trainloader)
        if self.checkpoint is None:
            self.checkpoint = Checkpoint(save_per_epoch = int(num_epochs/10)+1)

        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.trainloader)/10)
     
        print('===========================START TRAINING=================================')      
        for epoch in range(self.num_epochs+1):
            
            self.epoch = epoch
            self.training_epoch()

            if self.evaluate_per_epoch != 0:
                if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                    self.evaluate_epoch()
                    
                
            if self.scheduler is not None:
                self.scheduler.step()
            if (epoch % self.checkpoint.save_per_epoch == 0 or epoch == num_epochs - 1):
                self.checkpoint.save(self.model, epoch = epoch)
        print("Training Completed!")

    def training_epoch(self):
        self.model.train()
        epoch_loss = 0
        running_loss = 0
        running_time = 0

        for i, batch in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            start_time = time.time()
            loss = self.model.training_step(batch)
            loss.backward()
            
            # 
            if self.clip_grad is not None:
                clip_gradient(self.optimizer, self.clip_grad)

            self.optimizer.step()
            end_time = time.time()

            epoch_loss += loss.item()
            running_loss += loss.item()
            running_time += end_time-start_time
            iters = len(self.trainloader)*self.epoch+i+1
            if iters % self.print_per_iter == 0:
                print("[{}|{}] [{}|{}] || Training loss: {:10.4f} || Time: {:10.4f} s".format(self.epoch, self.num_epochs, iters, self.num_iters, running_loss/ self.print_per_iter, running_time))
                self.logging({"Training Loss/Batch" : running_loss/ self.print_per_iter,})
                running_loss = 0
                running_time = 0
        self.logging({"Training Loss/Epoch" : epoch_loss / len(self.trainloader),})

    def inference_batch(self, testloader):
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in testloader:
                outputs = self.model.inference_step(batch)
                if isinstance(outputs, (list, tuple)):
                    for i in outputs:
                        results.append(i)
                else:
                    results = outputs
                break      
        return results

    def inference_item(self, img):
        self.model.eval()

        with torch.no_grad():
            outputs = self.model.inference_step({"imgs": img.unsqueeze(0)})      
        return outputs


    def evaluate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        metric_dict = {}
        print('=============================EVALUATION===================================')
   
        with torch.no_grad():
            for batch in tqdm(self.valloader):
                loss, metrics = self.model.evaluate_step(batch)
                epoch_loss += loss.item()
                metric_dict.update(metrics)
        self.model.reset_metrics()
        print()
        print("[{}|{}] || Validation results || Val Loss: {:10.5f} |".format(self.epoch, self.num_epochs,epoch_loss / len(self.valloader)), end=' ')
        for metric, score in metric_dict.items():
            print(metric +': ' + str(score), end = ' | ')
        print('==')
        print('==========================================================================')

        log_dict = {"Validation Loss/Epoch" : epoch_loss / len(self.valloader),}
        log_dict.update(metric_dict)
        self.logging(log_dict)
        
    


    def logging(self, logs):
        tags = [l for l in logs.keys()]
        values = [l for l in logs.values()]
        self.logger.write(tags= tags, values= values)



    def forward_test(self):
        self.model.eval()
        outputs = self.model.forward_test()
        print("Feed forward success, outputs's shape: ", outputs.shape)

    def __str__(self):
        s0 = "---------MODEL INFO----------------"
        s1 = "Model name: " + self.model.model_name
        s2 = f"Number of trainable parameters:  {self.model.trainable_parameters():,}"
       
        s3 = "Loss function: " + str(self.criterion)[:-2]
        s4 = "Optimizer: " + str(self.optimizer)
        s5 = "Training iterations per epoch: " + str(len(self.trainloader))
        s6 = "Validating iterations per epoch: " + str(len(self.valloader))
        return "\n".join([s0,s1,s2,s3,s4,s5,s6])

    def set_attribute(self, kwargs):
        self.checkpoint = None
        self.scheduler = None
        self.clip_grad = None
        self.logger = None
        self.evaluate_per_epoch = 1
        for i,j in kwargs.items():
            setattr(self, i, j)

        if self.logger is None:
            self.logger = Logger()