import mindspore
import numpy as np
from mindspore import Tensor, nn, ops


def compute_flops(module, inp, out):
    if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
        return compute_Conv2d_flops(module, inp, out)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
        return compute_BatchNorm2d_flops(module, inp, out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d, nn.MaxPool1d, nn.AvgPool1d, mindspore.ops.AdaptiveAvgPool2D)):
        return compute_Pool2d_flops(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.Sigmoid)):
        return compute_ReLU_flops(module, inp, out)
    elif isinstance(module, nn.Dense):
        return compute_Linear_flops(module, inp, out)
    else:
        print(f"[Flops]: {type(module).__name__} is not supported!")
        return 0
    pass


"""
consume the flops of conv
"""


def compute_Conv2d_flops(module: nn.Cell, inp: Tensor, out: Tensor):
    assert isinstance(module, nn.Conv2d)
    assert len(inp.shape) == 4 and len(inp.shape) == len(out.shape)

    batch_size = inp.shape[0]
    in_c = inp.shape[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.shape[1:]
    group = module.group

    filters_per_channel = out_c // group
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_BatchNorm2d_flops(module: nn.Cell, inp: Tensor, out: Tensor):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.shape) == 4 and len(inp.shape) == len(out.shape)
    # in_c, in_h, in_w = inp.shape[1:]
    batch_flops = np.prod(inp.shape)
    if module.requires_grad:
        batch_flops *= 2
    return batch_flops


def compute_ReLU_flops(module: nn.Cell, inp: Tensor, out: Tensor):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU, ops.ReLU))
    batch_size = inp.shape[0]
    active_elements_count = batch_size

    for s in inp.shape[1:]:
        active_elements_count *= s

    return active_elements_count


def compute_Pool2d_flops(module: nn.Cell, inp: Tensor, out: Tensor):
    assert isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)
    assert len(inp.shape) == 4 and len(inp.shape) == len(out.shape)
    return np.prod(inp.shape)


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Dense)
    assert len(inp.shape) == 2 and len(out.shape) == 2
    batch_size = inp.shape[0]
    return batch_size * inp.shape[1] * out.shape[1]


if __name__ == '__main__':
    def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         has_bias=has_bias, pad_mode=pad_mode)


    channel = 3

    # net = conv(channel, 64, 11, stride=4, pad_mode="same", has_bias=True)
    net = conv(64, 128, 5, pad_mode="same", has_bias=True)
    input = np.ones([3, 244, 244]).reshape((1, 3, 244, 244))
    input = Tensor(np.array(input), dtype=mindspore.int32)
    print(compute_Conv2d_flops(net, input, input))
