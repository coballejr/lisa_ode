import sympy as sp
from .lie_field import LieField

class Prolongation:

    def __init__(self, LF: LieField):

        x, u = LF.x, LF.u
        xix, etau = LF.xix, LF.etau

        # standardize symbol space
        self.x_og, self.u_og = x, u
        self.xix_og, self.etau_og = xix, etau

        self.x, self.u = sp.symbols('x, u')
        self.xix = xix.subs({x: self.x, u: self.u})
        self.etau = etau.subs({x: self.x, u: self.u})

        self.ufunc = sp.Function(str(u))(self.x)

    def prolong(self, partials: str) -> sp.Expr:

        assert set(partials) <= set('x'), 'partials must be string of x.'

        ucoeff = self.ufunc
        eta = self.etau.subs({self.u: ucoeff}) # u -> ufunc for total derivative

        x =  self.x
        xix = self.xix
        xix = xix.subs({self.u: ucoeff}) # u -> ufunc for total derivative

        for var in partials:
            eta = eta.diff(x) - (ucoeff.diff(x)*xix.diff(x))
            ucoeff = ucoeff.diff(x)

        eta = eta.subs({x: self.x_og})

        return eta
