# Batch-Instance-Normalization

This repository provides an example of using [Batch-Instance Normalization (to appear NIPS 2018)](https://arxiv.org/abs/1805.07925) for classification on CIFAR-10/100, written by [Hyeonseob Nam](https://www.linkedin.com/in/hyeonseob-nam/) and [Hyo-Eun Kim](https://www.linkedin.com/in/hekim0530/) at [Lunit Inc.](https://lunit.io/)

Acknowledgement: This code is based on [Wei Yang's pytorch-classification](https://github.com/bearpaw/pytorch-classification)

## Citation
If you use this code for your research, please cite:
```
@article{nam2018batch,
  title={Batch-Instance Normalization for Adaptively Style-Invariant Neural Networks},
  author={Nam, Hyeonseob and Kim, Hyo-Eun},
  journal={arXiv preprint arXiv:1805.07925},
  year={2018}
}
```

## Prerequisites
- [PyTorch 0.4.0](https://pytorch.org/)
- Python >= 3.5
- Cuda >= 8.0

## Training Examples
Training ResNet-50 on CIFAR-100 using **Batch Normalization**
```
python main.py --dataset cifar100 --depth 50 --norm bn --checkpoint checkpoints/cifar100-resnet50-bn
```
Training ResNet-50 on CIFAR-100 using **Instance Normalization**
```
python main.py --dataset cifar100 --depth 50 --norm in --checkpoint checkpoints/cifar100-resnet50-in
```
Training ResNet-50 on CIFAR-100 using **Batch-Instance Normalization**
```
python main.py --dataset cifar100 --depth 50 --norm bin --checkpoint checkpoints/cifar100-resnet50-bin
```


## Summary of Results
### Classification on CIFAR-10/100 (ResNet-110) and ImageNet (ResNet-18)
<p align="center">
  <img src="figs/image_classification.png?raw=true" height="105"/>
</p>

### Classification on CIFAR-100 with different architectures
<p align="center">
  <img src="?raw=true"/>
</p>

### Mixed-domain classification on Office-Home (ResNet-18)
<p align="center">
  <img src="?raw=true"/>
</p>

### Character recognition
<p align="center">
  <img src="?raw=true"/>
</p>

## Links
- Tensorflow implementation by @taki0112: [code](https://github.com/taki0112/Batch_Instance_Normalization-Tensorflow)
