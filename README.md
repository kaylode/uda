# Unsupervised Data Augmentation for Image Classification

## Configuration for custom dataset:
Open file configs/configs.yaml
```
settings:
  project_name: <dataset's name> (name of the folder of the dataset that under ./data folder)
  train_imgs: train
  val_imgs: val
  test_imgs: test
```

## Training:
```
python train.py 
```

## Inference:
```
python detect.py --weight=<weight path>
```

## Reference:
- timm models from https://github.com/rwightman/pytorch-image-models
