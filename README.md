# keras-MDnet
Keras implementation of Multi-Domain Object Tracking

Officail [matlab implementation](https://github.com/HyeonseobNam/MDNet)

Officail [pytorch implementation](https://github.com/HyeonseobNam/py-MDNet)

Original [paper](https://arxiv.org/abs/1510.07945)

## Prerequisites
```
keras
scikit-learn (for Ridge regression)
```
### Usage
* Download [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html) or [VOT](http://www.votchallenge.net/) datasets
* Run the following command with the sequence name
```
 python mdnet_run.py -s "sequence" -r
```
### Details
Both matlab and torch models are pretrained on VOT/OTB datasets

This keras implementation converts torch model (trained only on VOT) to keras

Fine tuning and object tracking are done in keras backend

### TODO
* Implement pre-training scheme using keras
* Train keras model on both VOT and OTB
