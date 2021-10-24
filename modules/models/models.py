import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseTimmModel(nn.Module):
    """Some Information about BaseTimmModel"""

    def __init__(
        self,
        num_classes=1000,
        feat_dim = None,
        name="vit_base_patch16_224",
        from_pretrained=True
    ):
        super().__init__()
        self.name = name
        if num_classes != 1000:
            self.model = timm.create_model(name, pretrained=from_pretrained, num_classes=num_classes)
        else:
            self.model = timm.create_model(name, pretrained=from_pretrained)

        dim_in = self.model.num_features

        if feat_dim is not None:
            self.embedding_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

        self.model = nn.DataParallel(self.model)

    def forward(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs = self.model(inputs)
        return outputs

    def forward_unsup(self, batch, device):
        inputs = batch["imgs"]
        aug_inputs = batch["aug_imgs"]
        merged_inputs = torch.cat([inputs, aug_inputs], dim=0)
        merged_inputs = merged_inputs.to(device)

        features = self.model.forward_features(merged_inputs)
        features = F.normalize(self.embedding_head(features), dim=1)

        bsz = inputs.shape[0]
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        outputs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        return outputs





    