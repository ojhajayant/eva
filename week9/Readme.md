
Please refer [this notebook](https://github.com/ojhajayant/eva/blob/master/week9/S9_assignment.ipynb) for the code.

final accuracy of the model = 88.03%

Logs:
```
 0%|                                                                                          | 0/782 [00:00<?, ?it/s]
Model training starts on CIFAR10 dataset
EPOCH: 1
Loss=1.1938164234161377 Batch_id=781 Accuracy=49.78: 100%|███████████████████████████| 782/782 [02:01<00:00,  6.42it/s]

Test set: Average loss: 1.0116, Accuracy: 6326/10000 (63.26%)

validation-accuracy improved from 0 to 63.26, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-63.26.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 2
Loss=1.1422334909439087 Batch_id=781 Accuracy=68.65: 100%|███████████████████████████| 782/782 [02:03<00:00,  6.33it/s]

Test set: Average loss: 0.8019, Accuracy: 7297/10000 (72.97%)

validation-accuracy improved from 63.26 to 72.97, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-72.97.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 3
Loss=0.8033443689346313 Batch_id=781 Accuracy=76.36: 100%|███████████████████████████| 782/782 [02:03<00:00,  6.34it/s]

Test set: Average loss: 0.6432, Accuracy: 7740/10000 (77.40%)

validation-accuracy improved from 72.97 to 77.4, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-3_L1-1_L2-0_val_acc-77.4.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 4
Loss=0.4780978262424469 Batch_id=781 Accuracy=80.47: 100%|███████████████████████████| 782/782 [02:03<00:00,  6.32it/s]

Test set: Average loss: 0.5809, Accuracy: 8055/10000 (80.55%)

validation-accuracy improved from 77.4 to 80.55, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-4_L1-1_L2-0_val_acc-80.55.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 5
Loss=0.44373440742492676 Batch_id=781 Accuracy=83.43: 100%|██████████████████████████| 782/782 [02:04<00:00,  6.30it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.7568, Accuracy: 7516/10000 (75.16%)

EPOCH: 6
Loss=0.4987477958202362 Batch_id=781 Accuracy=85.56: 100%|███████████████████████████| 782/782 [02:03<00:00,  6.32it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.6070, Accuracy: 7960/10000 (79.60%)

EPOCH: 7
Loss=0.4333514869213104 Batch_id=781 Accuracy=87.53: 100%|███████████████████████████| 782/782 [02:03<00:00,  6.34it/s]

Test set: Average loss: 0.5947, Accuracy: 8124/10000 (81.24%)

validation-accuracy improved from 80.55 to 81.24, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-7_L1-1_L2-0_val_acc-81.24.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 8
Loss=0.7081902027130127 Batch_id=781 Accuracy=89.11: 100%|███████████████████████████| 782/782 [02:03<00:00,  6.34it/s]

Test set: Average loss: 0.5526, Accuracy: 8277/10000 (82.77%)

validation-accuracy improved from 81.24 to 82.77, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-8_L1-1_L2-0_val_acc-82.77.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 9
Loss=0.6426583528518677 Batch_id=781 Accuracy=90.02: 100%|███████████████████████████| 782/782 [02:03<00:00,  6.34it/s]

Test set: Average loss: 0.4893, Accuracy: 8533/10000 (85.33%)

validation-accuracy improved from 82.77 to 85.33, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-9_L1-1_L2-0_val_acc-85.33.h5
EPOCH: 10
Loss=0.1453838050365448 Batch_id=781 Accuracy=91.37: 100%|███████████████████████████| 782/782 [02:03<00:00,  6.33it/s]

Test set: Average loss: 0.4644, Accuracy: 8548/10000 (85.48%)

validation-accuracy improved from 85.33 to 85.48, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-10_L1-1_L2-0_val_acc-85.48.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 11
Loss=0.09714078903198242 Batch_id=781 Accuracy=92.27: 100%|██████████████████████████| 782/782 [02:03<00:00,  6.32it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5025, Accuracy: 8470/10000 (84.70%)

EPOCH: 12
Loss=0.39840278029441833 Batch_id=781 Accuracy=93.20: 100%|██████████████████████████| 782/782 [02:04<00:00,  6.27it/s]

Test set: Average loss: 0.4585, Accuracy: 8597/10000 (85.97%)

validation-accuracy improved from 85.48 to 85.97, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-12_L1-1_L2-0_val_acc-85.97.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 13
Loss=0.5929864048957825 Batch_id=781 Accuracy=94.22: 100%|███████████████████████████| 782/782 [02:04<00:00,  6.28it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5487, Accuracy: 8429/10000 (84.29%)

EPOCH: 14
Loss=0.49649715423583984 Batch_id=781 Accuracy=94.54: 100%|██████████████████████████| 782/782 [02:03<00:00,  6.31it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5354, Accuracy: 8492/10000 (84.92%)

EPOCH: 15
Loss=0.06074455380439758 Batch_id=781 Accuracy=95.28: 100%|██████████████████████████| 782/782 [02:04<00:00,  6.30it/s]

Test set: Average loss: 0.4947, Accuracy: 8627/10000 (86.27%)

validation-accuracy improved from 85.97 to 86.27, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-15_L1-1_L2-0_val_acc-86.27.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 16
Loss=0.11008298397064209 Batch_id=781 Accuracy=95.85: 100%|██████████████████████████| 782/782 [02:04<00:00,  6.28it/s]

Test set: Average loss: 0.4586, Accuracy: 8698/10000 (86.98%)

validation-accuracy improved from 86.27 to 86.98, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-16_L1-1_L2-0_val_acc-86.98.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 17
Loss=0.16396388411521912 Batch_id=781 Accuracy=96.25: 100%|██████████████████████████| 782/782 [02:05<00:00,  6.23it/s]

Test set: Average loss: 0.4785, Accuracy: 8745/10000 (87.45%)

validation-accuracy improved from 86.98 to 87.45, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-17_L1-1_L2-0_val_acc-87.45.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 18
Loss=0.062398552894592285 Batch_id=781 Accuracy=96.66: 100%|█████████████████████████| 782/782 [04:06<00:00,  3.17it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.6023, Accuracy: 8491/10000 (84.91%)

EPOCH: 19
Loss=0.23561334609985352 Batch_id=781 Accuracy=96.85: 100%|██████████████████████████| 782/782 [02:08<00:00,  6.10it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5843, Accuracy: 8595/10000 (85.95%)

EPOCH: 20
Loss=0.016510009765625 Batch_id=781 Accuracy=97.15: 100%|████████████████████████████| 782/782 [02:11<00:00,  5.95it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5118, Accuracy: 8743/10000 (87.43%)

EPOCH: 21
Loss=0.2903893291950226 Batch_id=781 Accuracy=97.44: 100%|███████████████████████████| 782/782 [02:12<00:00,  5.88it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5132, Accuracy: 8715/10000 (87.15%)

EPOCH: 22
Loss=0.10776406526565552 Batch_id=781 Accuracy=97.42: 100%|██████████████████████████| 782/782 [02:12<00:00,  5.91it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5427, Accuracy: 8692/10000 (86.92%)

EPOCH: 23
Loss=0.49669167399406433 Batch_id=781 Accuracy=97.87: 100%|██████████████████████████| 782/782 [02:13<00:00,  5.86it/s]

Test set: Average loss: 0.4945, Accuracy: 8763/10000 (87.63%)

validation-accuracy improved from 87.45 to 87.63, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-23_L1-1_L2-0_val_acc-87.63.h5
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
EPOCH: 24
Loss=0.3561747670173645 Batch_id=781 Accuracy=98.00: 100%|███████████████████████████| 782/782 [02:14<00:00,  5.84it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5139, Accuracy: 8728/10000 (87.28%)

EPOCH: 25
Loss=0.3673626780509949 Batch_id=781 Accuracy=98.02: 100%|███████████████████████████| 782/782 [02:15<00:00,  5.79it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5410, Accuracy: 8730/10000 (87.30%)

EPOCH: 26
Loss=0.04883211851119995 Batch_id=781 Accuracy=98.28: 100%|██████████████████████████| 782/782 [02:15<00:00,  5.77it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5601, Accuracy: 8701/10000 (87.01%)

EPOCH: 27
Loss=0.1723727136850357 Batch_id=781 Accuracy=98.34: 100%|███████████████████████████| 782/782 [02:13<00:00,  5.88it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5934, Accuracy: 8713/10000 (87.13%)

EPOCH: 28
Loss=0.19442865252494812 Batch_id=781 Accuracy=98.52: 100%|██████████████████████████| 782/782 [02:11<00:00,  5.95it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5288, Accuracy: 8755/10000 (87.55%)

EPOCH: 29
Loss=0.14253392815589905 Batch_id=781 Accuracy=98.36: 100%|██████████████████████████| 782/782 [02:13<00:00,  5.84it/s]
  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.5478, Accuracy: 8733/10000 (87.33%)

EPOCH: 30
Loss=0.019111186265945435 Batch_id=781 Accuracy=98.72: 100%|█████████████████████████| 782/782 [02:13<00:00,  5.85it/s]

Test set: Average loss: 0.5440, Accuracy: 8803/10000 (88.03%)

validation-accuracy improved from 87.63 to 88.03, saving model to D:\PG-ML\eva4\week9\./saved_models/CIFAR10_model_epoch-30_L1-1_L2-0_val_acc-88.03.h5
```

The ResNet18 model has been taken from [this link](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py).
