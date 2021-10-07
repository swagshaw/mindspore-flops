from collections import OrderedDict

import numpy as np
from mindspore import nn, Tensor, Parameter, numpy
import mindspore
from mindspore.ops import operations as ops
from msflops.compute_flops import compute_flops
from model_zoo.official.cv.resnet.src.resnet import resnet50, resnet18
from model_zoo.official.cv.densenet.src.network.densenet import _densenet121
from mindspore.nn import Cell
from typing import TypeVar, Callable

T = TypeVar('T', bound='Cell')

zeros = ops.Zeros()
size = ops.Size()


def apply(self: T, fn: Callable[['Cell'], None]) -> T:
    for module in self.cells():
        module.apply(fn)
    fn(self)
    return self


class ModelHook(object):
    def __init__(self, model, input_size):
        assert isinstance(model, nn.Cell)
        assert isinstance(input_size, (list, tuple))
        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook
        Cell.apply = apply
        self._hook_model()
        x = np.random.rand(1, *self._input_size)
        self._model.training = False
        self._model(Tensor(x, dtype=mindspore.float32))

    @staticmethod
    def _register_buffer(module: nn.Cell):
        assert isinstance(module, nn.Cell)
        if len(list(module.cells())) > 0:
            return
        module.insert_param_to_cell('Flops', Parameter(Tensor(np.zeros(1), dtype=mindspore.int64)))
        # print(module.cls_name)
        # print(module.Flops)

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        # print(self._model.cells())
        self._sub_module_call_hook()

    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call

            output = self._origin_call[module.__class__](module, *input, **kwargs)
            if len(input) == 1:
                flops = compute_flops(module, input[0], output)
            elif len(input) > 1:
                flops = compute_flops(module, input, output)
            else:  # error
                flops = 0
            module.Flops = Tensor(
                np.array([flops], dtype=np.int64))
            return output

        for module in self._model.cells():
            print(module)
            if isinstance(module,nn.SequentialCell):
                for item in module.cells():
                    if len(list(item.cells())) == 0 and item.__class__ not in self._origin_call:
                        self._origin_call[item.__class__] = item.__class__.__call__
                        item.__class__.__call__ = wrap_call
            if len(list(module.cells())) == 0 and module.__class__ not in self._origin_call:
                # print(module.cells())
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    @staticmethod
    def _retrieve_leaf_modules(model):
        leaf_modules = []
        for name, m in model.cells_and_names():
            if len(list(m.cells())) == 0:
                leaf_modules.append((name, m))
        return leaf_modules

    def retrieve_leaf_modules(self):
        return OrderedDict(self._retrieve_leaf_modules(self._model))


if __name__ == '__main__':
    modelhook = ModelHook(_densenet121(), (3, 224, 224))
    print(modelhook._hook_model())
