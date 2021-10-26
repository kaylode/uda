import torch
import torch.nn as nn


class UDALoss(nn.Module):
    def __init__(self, beta=0., temperature=1., lamb=0.5):
        self.beta = beta
        self.lamb = lamb
        self.temperature = temperature
        self.sup_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.unsup_criterion = nn.KLDivLoss(reduction='none')

    def forward(self, sup_outputs, targets, unsup_outputs=None, unsup_outputs_aug=None):
        sup_loss = self.sup_criterion(sup_outputs, targets)

        if unsup_outputs is not None and unsup_outputs_aug is not None:
            unsup_outputs = unsup_outputs.detach()
            unsup_y_probas = torch.softmax(unsup_outputs, dim=-1).detach()

            # confidence-based masking
            if self.beta != 0:
                unsup_loss_mask = torch.max(unsup_y_probas, dim=-1)[0] > self.beta
                unsup_loss_mask = unsup_loss_mask.float()
            else:
                unsup_loss_mask = torch.ones(unsup_outputs.shape[0], dtype=torch.float32)

            num_unsup = torch.sum(unsup_loss_mask, dim=-1)

            if num_unsup == 0.:
                unsup_loss = torch.FloatTensor([0.])
                loss = sup_loss            
            else:

                # temperature scaling
                unsup_aug_y_probas = torch.log_softmax(unsup_outputs_aug/self.temperature, dim=-1)
                unsup_loss = self.unsup_criterion(unsup_aug_y_probas, unsup_y_probas)
                unsup_loss = torch.sum(unsup_loss, dim=-1)
                
                unsup_loss_mask = unsup_loss_mask.to(self.device)
                unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.sum(unsup_loss_mask, dim=-1)
                loss = sup_loss + self.lamb * unsup_loss
        else:
            loss = sup_loss
            unsup_loss = torch.FloatTensor([0.])

        return loss, {
            'T': loss.item(),
            'UNSUP': unsup_loss.item(),
            'SUP': sup_loss.item()
        }