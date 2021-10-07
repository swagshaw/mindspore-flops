# msflops

This script is designed to compute the theoretical amount of multiply-add operations
in convolutional neural networks. It also can compute the number of parameters(TODO) and
print per-layer computational cost of a given network.

Supported layers:
- Conv1d/2d/3d (including grouping)
- BatchNorm1d/2d/3d, GroupNorm, InstanceNorm1d/2d/3d
- Activations (ReLU, PReLU, ELU, ReLU6, LeakyReLU)
- Dense
- Poolings (AvgPool1d/2d/3d, MaxPool1d/2d/3d and adaptive ones)

## Example
```python
import sys
import mindspore
from msflops import stat
from models import alexnet
from mindspore import context

if __name__ == '__main__':
    model = alexnet.AlexNet() # which is a model designed by yourself
    context.set_context(mode=context.PYNATIVE_MODE)
    stat(model, (3, 224, 224))
```

## Benchmark

TODO:Fill a table about the data of more models in mindspore.modelzoo

Model         | Input Resolution | Flops 
---           |---               |---      
alexnet       |224x224           | 710.6(M)   
resnet18      |224x224           | 1.82(G)
resnet34      |224x224           | 3.67(G)
resnet50      |224x224           | 4.11(G)
resnet101      |224x224           | 7.83(G)
resnet152      |224x224           | 11.55(G)
vgg16          |224x224           | 15.49(G)
vgg16_bn          |224x224           | 15.51(G)
vgg11           |224x224           | 7.62(G)
vgg11_bn           |224x224           | 7.63(G)
vgg13           |224x224           | 11.32(G)
vgg13_bn           |224x224           | 11.34(G)
vgg19           |224x224           | 19.65(G)
vgg19_bn           |224x224           | 19.66(G)
densenet121       |224x224           | 2.87(G)
densenet161       |224x224           | 7.79(G)
densenet129       |224x224           | 3.4(G)
densenet201       |224x224           | 4.34(G)
squeezenet         |224x224           | 715(M)
squeezenet_res         |224x224           | 711.5(M)
shufflenetv1|224x224           | 536.78(M)
shufflenetv2|224x224           | 149.6(M)
efficientnetb0|224x224           | 392.55(M)