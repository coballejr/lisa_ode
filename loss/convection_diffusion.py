from .grad_graph import GradCollection

class ConvDiff:

    def __init__(self, v:float, k:float = 1.0):
        self.v = v
        self.k = k

    def get_grads(self):
        return ['u_x','u_xx']

    def __call__(self, grads:GradCollection):

        u_x = grads.u_x
        u_xx = grads.u_xx

        f_u = self.v*u_x - self.k*u_xx

        return f_u
