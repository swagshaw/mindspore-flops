# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Alexnet."""
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     has_bias=has_bias, pad_mode=pad_mode)


def fc_with_initialize(input_channels, out_channels, has_bias=True):
    return nn.Dense(input_channels, out_channels, has_bias=has_bias)


class AlexNet(nn.Cell):
    """
    Alexnet
    """

    def __init__(self, num_classes=10, channel=3, phase='train', include_top=True):
        super(AlexNet, self).__init__()
        self.conv1 = conv(channel, 64, kernel_size=11, stride=4, padding=2, pad_mode="pad", has_bias=True)
        self.conv2 = conv(64, 192, kernel_size=5, padding=2, pad_mode="pad",has_bias=True)
        self.conv3 = conv(192, 384, kernel_size=3, padding=1,pad_mode="pad", has_bias=True)
        self.conv4 = conv(384, 256, kernel_size=3,   padding=1,pad_mode="pad", has_bias=True)
        self.conv5 = conv(256, 256, kernel_size=3,  padding=1,pad_mode="pad", has_bias=True)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0
            self.flatten = nn.Flatten()
            self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
            self.fc2 = fc_with_initialize(4096, 4096)
            self.fc3 = fc_with_initialize(4096, num_classes)
            self.dropout = nn.Dropout(dropout_ratio)
        context.set_context(mode=context.PYNATIVE_MODE)
    def construct(self, x):
        """define network"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    net = AlexNet()
    print(net.name_cells())