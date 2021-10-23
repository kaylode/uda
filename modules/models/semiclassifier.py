from .base_model import BaseModel
import torch
import sys
sys.path.append('..')

class SemiClassifier(BaseModel):
    def __init__(self, model, sup_criterion=None, unsup_criterion=None, **kwargs):
        super(SemiClassifier, self).__init__(**kwargs)
        self.model = model
        self.model_name = self.model.name
        self.sup_criterion = sup_criterion
        self.unsup_criterion = unsup_criterion
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
            self.set_optimizer_params()

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, unsup_batch):
        # Supervised pipeline
        outputs = self.model(batch, self.device)
        targets =  batch['targets'].to(self.device)
        loss = self.sup_criterion(outputs, targets)

        #Unsupervised pipeline
        unsup_outputs = self.model.forward_unsup(unsup_batch, self.device)
        unsup_loss = self.unsup_criterion(unsup_outputs)

        total_loss = 0.8 * loss + 0.2 * unsup_loss
        loss_dict = {'SUP': loss.item(), 'UNSUP':unsup_loss.item(), 'T': total_loss.item()}
        return total_loss, loss_dict

    def inference_step(self, batch, return_probs=False):
        outputs = self.model(batch, self.device)
        preds = torch.argmax(outputs, dim=1)
        preds = preds.detach()
        if return_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs, _ = torch.max(probs, dim=1)
            return preds.cpu().numpy(), probs.cpu().numpy()
        else:
            return preds.numpy()

    def evaluate_step(self, batch):
        # Supervised pipeline
        outputs = self.model(batch, self.device)
        targets =  batch['targets'].to(self.device)
        loss = self.sup_criterion(outputs, targets)

        #Unsupervised pipeline
        # unsup_outputs = self.model.forward_unsup(unsup_batch, self.device)
        # unsup_loss = self.unsup_criterion(unsup_outputs)

        # total_loss = 0.8 * loss + 0.2 * unsup_loss
        loss_dict = {'T': loss.item()}

        self.update_metrics(outputs = outputs, targets = targets)
        return loss, loss_dict