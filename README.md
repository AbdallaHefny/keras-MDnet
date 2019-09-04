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
* Run 
```
   python mdnet_run.py -s Car1 -r
```
* You may replace 'Car1' with any [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html) or [VOT](http://www.votchallenge.net/) sequence
### Details
Both matlab and torch models are pretrained on VOT/OTB datasets

This keras implementation converts torch model (trained only on VOT) to keras

Fine tuning, hard negative mining, and object tracking are done in using keras 

### TODO
* Remove dependency on tensorflow to do Local Response Normalization
* Implement pre-training scheme using keras
* Train keras model on both VOT and OTB
