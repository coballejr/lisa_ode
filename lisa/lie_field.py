import sympy as sp
from typing import Tuple

class LieField:

    def __init__(self, x: sp.Symbol,
                     u: sp.Symbol,
                     xix: sp.Expr,
                     etau: sp.Expr
                     ):

        assert xix.free_symbols <= set([x, u]), 'xix can only depend on x and u.'
        assert etau.free_symbols <= set([x, u]), 'etau can only depend on x and u.'

        self.x, self.u = x, u
        self.xix = xix
        self.etau = etau

    def X(self, xstar: sp.Expr,
              ustar: sp.Expr) -> Tuple[sp.Expr]:

        x, u, xix, etau = self.x, self.u, self.xix, self.etau

        xstar = xix*xstar.diff(x) + etau*xstar.diff(u)
        ustar = xix*ustar.diff(x) + etau*ustar.diff(u)

        return xstar, ustar

    def infs_to_np(self):
        x, u = self.x, self.u
        xix, etau = self.xix, self.etau

        xix_np = sp.lambdify([x,u], xix, 'numpy')
        etau_np = sp.lambdify([x,u], etau, 'numpy')

        return xix_np, etau_np


# Convection-diffusion continuous symmetries
v, k = 50, 1

class X1(LieField):

    def __init__(self):
        x, u = sp.symbols('x, u')
        xix = sp.sympify(0)
        etau = -0.25*((x**2)/(2*k))*u
        super(X1, self).__init__(x, u, xix, etau)

class X2(LieField):

    def __init__(self):
        x, u = sp.symbols('x, u')
        xix =  0.5*x
        etau = sp.sympify(0)
        super(X2, self).__init__(x, u, xix, etau)

class X3(LieField):

    def __init__(self):
        x, u = sp.symbols('x, u')
        xix = -(1/k)*sp.exp(-v*x/k)
        etau = -(v*u/k)*(sp.exp(-v*x/k) + sp.exp(v*x/k))
        super(X3, self).__init__(x, u, xix, etau)

class X4(LieField):

    def __init__(self):
        x, u = sp.symbols('x, u')
        xix =  sp.sympify(0)
        etau = x*u
        super(X4, self).__init__(x, u, xix, etau)

class X5(LieField):

    def __init__(self):
        x, u = sp.symbols('x, u')
        xix =  sp.sympify(0)
        etau = sp.exp((1/2*k)*(v-sp.sqrt(v**2 + 4*v*k))*x)
        super(X5, self).__init__(x, u, xix, etau)

class X6(LieField):

    def __init__(self):
        x, u = sp.symbols('x, u')
        xix =  sp.sympify(0)
        etau = sp.sympify(1)
        super(X6, self).__init__(x, u, xix, etau)

if __name__ == '__main__':
    import numpy as np
    lf = X5()
    xix, etau = lf.infs_to_np()
    input = np.random.rand(5,2)
    x, u = input[:,0], input[:,1]
    output_xix = xix(x, u)
    output_etau = etau(x, u)

    print(output_xix)
    print(output_etau)


