import os
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from .checkpoint import Checkpoint
import numpy as np
from tools.loggers.loggers import Logger
import time
from tools.utils.cuda import NativeScaler
from torch.cuda import amp
from tools.utils.gradcam import GradCam, show_cam_on_image
from datasets.augmentations.transforms import Denormalize


class Trainer():
    def __init__(self,
                 config,
                 model,
                 trainloader,
                 valloader,
                 **kwargs):

        self.cfg = config
        self.model = model
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.trainloader = trainloader
        self.valloader = valloader
        self.metrics = model.metrics  # list of metrics
        self.set_attribute(kwargs)

    def fit(self, start_epoch=0, start_iter=0, num_epochs=10, print_per_iter=None):
        self.num_epochs = num_epochs
        self.num_iters = (num_epochs+1) * len(self.trainloader)
        if self.checkpoint is None:
            self.checkpoint = Checkpoint(save_per_epoch=int(num_epochs/10)+1)

        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.trainloader)/10)

        self.epoch = start_epoch

        # For one-cycle lr only
        if self.scheduler is not None and self.step_per_epoch:
            self.scheduler.last_epoch = start_epoch - 1

        self.start_iter = start_iter % len(self.trainloader)
        self.iters = self.start_iter + len(self.trainloader)*self.epoch + 1

        print(f'===========================START TRAINING=================================')
        for epoch in range(self.epoch, self.num_epochs):
            try:
                self.epoch = epoch
                
                if self.unsup_loader is None:
                    self.training_epoch()
                else:
                    self.semisup_training_epoch()

                if self.evaluate_per_epoch != 0:
                    if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                        self.evaluate_epoch()

                if self.scheduler is not None and self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Training/Learning rate': lr}
                    self.logging(log_dict, step=self.epoch)

            except KeyboardInterrupt:
                self.checkpoint.save(
                    self.model,
                    save_mode='last',
                    epoch=self.epoch,
                    iters=self.iters,
                    best_value=self.best_value,
                    class_names=self.trainloader.dataset.classes,
                    config=self.cfg)
                print("Stop training, checkpoint saved...")
                break

        print("Training Completed!")

    def semisup_training_epoch(self):
        sup_iter = iter(self.trainloader)
        self.model.train()

        running_loss = {}
        running_time = 0

        self.optimizer.zero_grad()
        for i, unsup_batch in enumerate(self.unsup_loader):
            try:
                batch = next(sup_iter)
            except StopIteration:
                sup_iter = iter(self.trainloader)
                batch = next(sup_iter)

            start_time = time.time()

            with amp.autocast(enabled=self.use_amp):
                loss, loss_dict = self.model.training_step(batch, unsup_batch)
                if self.use_accumulate:
                    loss /= self.accumulate_steps

            self.model.scaler(loss, self.optimizer)

            if self.use_accumulate:
                if (i+1) % self.accumulate_steps == 0 or i == len(self.unsup_loader)-1:
                    self.model.scaler.step(
                        self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                    self.optimizer.zero_grad()

                    if self.scheduler is not None and not self.step_per_epoch:
                        self.scheduler.step()
                        lrl = [x['lr'] for x in self.optimizer.param_groups]
                        lr = sum(lrl) / len(lrl)
                        log_dict = {'Training/Learning rate': lr}
                        self.logging(log_dict, step=self.iters)
            else:
                self.model.scaler.step(
                    self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                self.optimizer.zero_grad()
                if self.scheduler is not None and not self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Training/Learning rate': lr}
                    self.logging(log_dict, step=self.iters)

            torch.cuda.synchronize()

            end_time = time.time()

            for (key, value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time-start_time
            self.iters = self.start_iter + \
                len(self.unsup_loader)*self.epoch + i + 1
            if self.iters % self.print_per_iter == 0:

                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[
                    1:-1].replace("'", '').replace(",", ' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(
                    self.epoch, self.num_epochs, self.iters, self.num_iters, loss_string, running_time))
                self.logging(
                    {"Training/Batch Loss": running_loss['T'] / self.print_per_iter, }, step=self.iters)
                running_loss = {}
                running_time = 0

            if (self.iters % self.checkpoint.save_per_iter == 0 or self.iters == self.num_iters - 1):
                print(f'Save model at [{self.epoch}|{self.iters}] to last.pth')
                self.checkpoint.save(
                    self.model,
                    save_mode='last',
                    epoch=self.epoch,
                    iters=self.iters,
                    best_value=self.best_value,
                    config=self.cfg)

    def training_epoch(self):
        self.model.train()

        running_loss = {}
        running_time = 0

        self.optimizer.zero_grad()
        
        for i, batch in enumerate(self.trainloader):

            start_time = time.time()

            with amp.autocast(enabled=self.use_amp):
                loss, loss_dict = self.model.training_step(batch)
                if self.use_accumulate:
                    loss /= self.accumulate_steps

            self.model.scaler(loss, self.optimizer)

            if self.use_accumulate:
                if (i+1) % self.accumulate_steps == 0 or i == len(self.trainloader)-1:
                    self.model.scaler.step(
                        self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                    self.optimizer.zero_grad()

                    if self.scheduler is not None and not self.step_per_epoch:
                        self.scheduler.step()
                        lrl = [x['lr'] for x in self.optimizer.param_groups]
                        lr = sum(lrl) / len(lrl)
                        log_dict = {'Training/Learning rate': lr}
                        self.logging(log_dict, step=self.iters)
            else:
                self.model.scaler.step(
                    self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                self.optimizer.zero_grad()
                if self.scheduler is not None and not self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Training/Learning rate': lr}
                    self.logging(log_dict, step=self.iters)

            torch.cuda.synchronize()

            end_time = time.time()

            for (key, value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time-start_time
            
            self.iters = self.start_iter + \
                len(self.trainloader)*self.epoch + i + 1
            if self.iters % self.print_per_iter == 0:

                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[
                    1:-1].replace("'", '').replace(",", ' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(
                    self.epoch, self.num_epochs, self.iters, self.num_iters, loss_string, running_time))
                self.logging(
                    {"Training/Batch Loss": running_loss['T'] / self.print_per_iter, }, step=self.iters)
                running_loss = {}
                running_time = 0

            if (self.iters % self.checkpoint.save_per_iter == 0 or self.iters == self.num_iters - 1):
                print(f'Save model at [{self.epoch}|{self.iters}] to last.pth')
                self.checkpoint.save(
                    self.model,
                    save_mode='last',
                    epoch=self.epoch,
                    iters=self.iters,
                    best_value=self.best_value,
                    class_names=self.trainloader.dataset.classes,
                    config=self.cfg)

    def evaluate_epoch(self):
        self.model.eval()
        epoch_loss = {}

        metric_dict = {}
        print('=============================EVALUATION===================================')
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(self.valloader):
                loss, loss_dict = self.model.evaluate_step(batch)

                for (key, value) in loss_dict.items():
                    if key in epoch_loss.keys():
                        epoch_loss[key] += value
                    else:
                        epoch_loss[key] = value

        end_time = time.time()
        running_time = end_time - start_time

        if self.model.metrics is not None:
            metric_dict = self.model.get_metric_values()
            self.model.reset_metrics()

        for key in epoch_loss.keys():
            epoch_loss[key] /= len(self.valloader)
            epoch_loss[key] = np.round(epoch_loss[key], 5)
        loss_string = '{}'.format(epoch_loss)[
            1:-1].replace("'", '').replace(",", ' ||')
        print()
        print("[{}|{}] || {} || Time: {:10.4f} s".format(
            self.epoch, self.num_epochs, loss_string, running_time))

        for metric, score in metric_dict.items():
            print(metric + ': ' + str(score), end=' | ')
        print()
        print('==========================================================================')

        log_dict = {
            "Validation/Epoch Loss": epoch_loss['T'] / len(self.valloader), }
        metric_log_dict = {f"Validation/{k}": v for k,
                           v in metric_dict.items()}
        log_dict.update(metric_log_dict)
        self.logging(log_dict, step=self.epoch)

        if self.model.metrics is not None:

            # Save model gives best mAP score
            if metric_dict['acc'] > self.best_value:
                self.best_value = metric_dict['acc']
                self.checkpoint.save(
                    self.model,
                    save_mode='best',
                    epoch=self.epoch,
                    iters=self.iters,
                    best_value=self.best_value,
                    class_names=self.trainloader.dataset.classes,
                    config=self.cfg)

        if self.visualize_when_val:
            self.visualize_batch()

    def visualize_batch(self):
        # Vizualize Grad Class Activation Mapping
        if not os.path.exists('./samples'):
            os.mkdir('./samples')

        denom = Denormalize()
        batch = next(iter(self.valloader))
        images = batch["imgs"]

        self.model.eval()

        config_name = self.cfg.model_name.split('_')[0]
        grad_cam = GradCam(model=self.model.model, config_name=config_name)

        for idx, inputs in enumerate(images):
            image_outname = os.path.join(
                'samples', f'{self.epoch}_{self.iters}_{idx}.jpg')
            img_show = denom(inputs)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(self.model.device)
            target_category = None
            grayscale_cam, label_idx = grad_cam(inputs, target_category)
            label = self.cfg.obj_list[label_idx]
            img_cam = show_cam_on_image(img_show, grayscale_cam, label)
            cv2.imwrite(image_outname, img_cam)

        img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)
        
        fig_cam = plt.figure()
        plt.imshow(img_cam)

        self.logger.write_image(
            f'samples/{self.epoch}_{self.iters}_{idx}', fig_cam, step=self.epoch)

    def logging(self, logs, step):
        tags = [l for l in logs.keys()]
        values = [l for l in logs.values()]
        self.logger.write(tags=tags, values=values, step=step)

    def set_accumulate_step(self):
        self.use_accumulate = False
        if self.cfg.total_accumulate_steps > 0:
            self.use_accumulate = True
            self.accumulate_steps = max(
                round(self.cfg.total_accumulate_steps / self.cfg.batch_size), 1)

    def set_amp(self):
        self.use_amp = False
        if self.cfg.mixed_precision:
            self.use_amp = True

    def __str__(self):
        s0 = "##########   MODEL INFO   ##########"
        s1 = "Model name: " + self.model.model_name
        s2 = f"Number of trainable parameters:  {self.model.trainable_parameters():,}"

        s5 = "Training iterations per epoch: " + str(len(self.trainloader))
        s6 = "Validating iterations per epoch: " + str(len(self.valloader))
        return "\n".join([s0, s1, s2, s5, s6])

    def set_attribute(self, kwargs):
        self.unsup_loader = None
        self.checkpoint = None
        self.scheduler = None
        self.clip_grad = 10.0
        self.logger = None
        self.visualize_when_val = True
        self.step_per_epoch = False
        self.evaluate_per_epoch = 1
        self.best_value = 0.0
        self.set_accumulate_step()
        self.set_amp()
        for i, j in kwargs.items():
            setattr(self, i, j)

        if self.logger is None:
            self.logger = Logger()
