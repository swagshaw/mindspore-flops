import sys
import mindspore
from msflops import stat
from models import alexnet
from mindspore import context
ost = sys.stdout

if __name__ == '__main__':
    """
    AlexNet
    """
    model = alexnet.AlexNet()
    context.set_context(mode=context.PYNATIVE_MODE)
    stat(model, (3, 224, 224))
