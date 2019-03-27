## Multi-view CNN code analysis

### Indexing:
- [Overview](#Overview)
- [Train_mvcnn](#train_mvcnn)
- [SVCNN](#SVCNN)
- [MVCNN](#MVCNN)
- [References](#References)
---
### Overview
Component:
- train_mvcnn.py
- MVCNN.py
- Model.py

---
### Train_mvcnn
#### Parameters
- num_views: number of views

#### Main
**STAGE 1**
- SVCNN: model1 = SVCNN
- Train_dataset: train_dataset = SingleImgDataset
- Trainer: trainer = ModelNetTrainer

**STAGE 2**
- MVCNN: model2 = MVCNN
- Train_dataset: train_dataset = MultiviewImgDataset
- Trainer: trainer = ModelNetTrainer

---
### SVCNN
**Process**
- net: self.net = models.resnet18(pretrained=self.pretraining)
- fc: self.net.fc = nn.Linear(512, 40)

---
### MVCNN
**Process**
- net_1: self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
- max_pooling: torch.max(y, 1)[0]
- net_2: self.net_2 = model.net.fc

---
### References
- [Pytorch code](https://github.com/jongchyisu/mvcnn_pytorch)
- [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.pdf)
---
