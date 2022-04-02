'''
adopted from:
    https://github.com/NickGeneva/taylor_green_pinns/blob/main/loss/grad_graph.py
'''
import torch
from torch.autograd import grad
from typing import Iterable

class GradCollection(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def valid(self, required: Iterable):
        for name in required:
            if not hasattr(self, name):
                return False
        return True

    def keys(self):
        return 0

    def size(self):
        return 0

class GradGraph(object):

    def __init__(self,
        ind_var: Iterable[str],
        dep_var: Iterable[str]
    ):

        self.roots = dep_var
        self.indp = ind_var
        self.map = {root:[] for root in self.roots}

    def add_grad(self, *args:str):

        for grad in args:

            assert isinstance(grad, str), 'Gradients must symbolically added with strings'
            assert grad.count('_') == 1
            assert grad.split('_')[0] in self.roots

            root_var = grad.split('_')[0]
            deriv = grad.split('_')[1]

            # Stack of derivatives up to 0th order
            nstack = [root_var] + [grad[:(3+i)] for i in range(len(deriv))]
            child = []
            while len(nstack) > 0:
                var_name = nstack.pop(-1)
                # Add node to graph
                if not var_name in self.map:
                    self.map[var_name] = child
                    child = [var_name]
                # Derivative present in graph so append and break
                else:
                    self.map[var_name] = self.map[var_name] + child
                    break

    def add_pde(self, pde_obj):
        self.add_grad(*pde_obj.get_grads())

    def calc_grad(self, **kwargs):
        gcoll = GradCollection(**kwargs)
        # TODO: Updated required var list based on graph
        assert gcoll.valid(self.indp), "Missing independent variables"
        assert gcoll.valid(self.roots), "Missing dependent variables"

        stack = []
        for node in self.roots:
            stack = stack + [(child, node) for child in self.map[node]]
        # Breadth first search through derivative graph
        while len(stack) > 0:
            new_stack = []
            for node, parent in stack:
                cgrad = False
                if len(self.map[node]) > 0: # If children create graph to go deeper
                    cgrad = True

                # Calc derivative
                in_var = getattr(gcoll, parent)
                wrt_var = getattr(gcoll, node[-1])
                try:
                    deriv = grad(in_var.sum(), wrt_var, create_graph=True, retain_graph=True)[0]
                except RuntimeError as e:
                    print('Warning failed auto_grad: '+str(e)+' Setting grads to zero.')
                    deriv = torch.zeros_like(in_var)

                # Store derivative into collection
                setattr(gcoll, node, deriv)
                # Add any children to stack
                new_stack = new_stack + [(child, node) for child in self.map[node]]

            stack = new_stack

        return gcoll
