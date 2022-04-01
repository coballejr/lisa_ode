from .grad_graph import GradCollection

class SeperableODE:

    def __init__(self):
        return

    def get_grads(self):
        return ['u_x']

    def __call__(self, grads:GradCollection):

        u_x = grads.u_x
        u = grads.u

        f_u = u_x - u**2

        return f_u
