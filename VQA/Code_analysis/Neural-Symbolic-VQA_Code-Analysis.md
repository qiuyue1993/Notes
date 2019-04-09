## Neural-Symbolic Visual Question Answering (NS-VQA) Code Analysis

### Indexing
- [Steps](#Steps)
- [References](#References)

---
### Steps
The overall framework has three steps:
**Object detection**
- code:
```
python scene_parse/mask_rcnn/tools/test_net.py
```
- input: raw images
- output: object proposals including object class labels, masks and scores
- output file: clevr_val_pretrained/detections.pkl

**Attribute extraction**
- Preprocess code:
```
python scene_parse/attr_net/tools/process_proposals.py
```
- generates a .json file which can be loaded by the attribute network

- Attribute extraction code:
```
python scene_parse/attr_net/tools/run_test.py
```
- which will generate parsed scenes that are going to be used for reasoning.

**Reasoning**
- Code:
```
python reason/tools/run_test.py
```
- The pretrained model will yield an overall question accuracy on CLEVR_v1.0 test split.

### References
- [Code](https://github.com/kexinyi/ns-vqa)

---
