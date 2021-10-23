import torch.utils.data as data
from .dataset import ImageSet, UnsupImageSet
from datasets.augmentations.transforms import get_augmentation

class SupDataloader(data.DataLoader):
    def __init__(self, config, root_dir, type, batch_size):

        self.root_dir = root_dir
        self.dataset = ImageSet(
            root_dir=root_dir, 
            transforms=get_augmentation(config, _type=type)
        )

        self.collate_fn = self.dataset.collate_fn
        
        super(SupDataloader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn)


class UnsupDataloader(data.DataLoader):
    def __init__(self, config, root_dir, batch_size):
        self.root_dir = root_dir
        self.dataset = UnsupImageSet(
            root_dir=root_dir, 
            transforms=get_augmentation(config, _type='train'),
            unsup_transforms=get_augmentation(config, _type='val')
        )

        self.collate_fn = self.dataset.collate_fn
        
        super(SupDataloader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn)