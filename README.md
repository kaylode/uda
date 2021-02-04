# Unsupervised Data Augmentation

- Edit configs in supervised.yaml and unsupervised.yaml

## Train supervised on CIFAR-10:
```
python supervised.py 
```

## Train unsupervised on CIFAR-10:
```
python UDA.py --limit=10000
```

## Results: (Efficientnet-b0)
Method | Dataset | No. labeleds | No. unlabeleds| Accuracy
--- | --- | --- | --- | ---
Supervised | CIFAR-10 | 50000 | 0 | 0
UDA | CIFAR-10 | 10000 | 40000 | 0
