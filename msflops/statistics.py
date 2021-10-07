from collections import OrderedDict
from msflops import StatTree, StatNode, report_format, ModelHook
from mindspore import nn


def get_parent_node(root_node, stat_node_name):
    assert isinstance(root_node, StatNode)

    node = root_node
    names = stat_node_name.split('.')
    for i in range(len(names) - 1):
        node_name = '.'.join(names[0:i + 1])
        child_index = node.find_child_index(node_name)
        assert child_index != -1
        node = node.children[child_index]
    return node


def convert_leaf_modules_to_stat_tree(leaf_modules):
    assert isinstance(leaf_modules, OrderedDict)

    create_index = 1
    root_node = StatNode(name='root', parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        names = leaf_module_name.split('.')
        for i in range(len(names)):
            create_index += 1
            stat_node_name = '.'.join(names[0:i + 1])
            parent_node = get_parent_node(root_node, stat_node_name)
            node = StatNode(name=stat_node_name, parent=parent_node)
            parent_node.add_child(node)
            if i == len(names) - 1:  # leaf module itself
                # if not isinstance(node, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU)):
                node.Flops = leaf_module.Flops.asnumpy()[0]

    return StatTree(root_node)


class ModelStat(object):
    def __init__(self, model, input_size, query_granularity=1):
        assert isinstance(model, nn.Cell)
        # assert isinstance(input_size, (tuple, list)) and len(input_size) == 3
        self._model = model
        self._input_size = input_size
        self._query_granularity = query_granularity

    def _analyze_model(self):
        model_hook = ModelHook(self._model, self._input_size)
        leaf_modules = model_hook.retrieve_leaf_modules()
        stat_tree = convert_leaf_modules_to_stat_tree(leaf_modules)
        collected_nodes = stat_tree.get_collected_stat_nodes(self._query_granularity)
        return collected_nodes

    def show_report(self):
        collected_nodes = self._analyze_model()
        report = report_format(collected_nodes)
        print(report)


def stat(model, input_size, query_granularity=1):
    ms = ModelStat(model, input_size, query_granularity)
    ms.show_report()
