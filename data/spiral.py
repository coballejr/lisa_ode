from scipy.integrate import solve_ivp
import numpy as np
from typing import Tuple

class Spiral:

    def __init__(self):
        return

    def dynamics(self, x:float, y:np.array) -> np.array:
        r = np.sqrt(x**2 + y**2)
        return (x*r + y*(r**2-1))/(x*(r**2-1) - y*r)

    def solve_num(self, y0: np.array,
                    xspan: Tuple = (0, 1)) -> np.array:
        xeval = np.linspace(xspan[0], xspan[1], 256)
        soln = solve_ivp(fun = self.dynamics, t_span = xspan, y0 = y0, method =
                        'Radau', t_eval = xeval)
        x, y = soln.t, soln.y
        return x, y

    def solve_canon(self, c: float, nsamps:int = 128, rinit: float = 0.1,
                   rfinal: float = 0.99) -> np.array:
        r = np.linspace(rinit, rfinal, nsamps, endpoint = False)
        theta = 0.5*np.log((1 -r)/(1+r)) + c
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        return x, y


if __name__ == '__main__':
    ode = Spiral()
    y0 = 0.1*np.random.rand(10) + 1
    x, theta = ode.solve_num(y0)

    import matplotlib.pyplot as plt
    for traj in theta:
        plt.plot(x, traj)
    plt.gca().set_aspect('equal')
    plt.show()
    plt.close()

    cs = np.array([0, 1.1, -1.1, 0.5])
    for c in cs:
        xtrue, ytrue = ode.solve_canon(c)
        plt.plot(xtrue, ytrue)
    plt.gca().set_aspect('equal')
    plt.show()
    plt.close()

    import sympy as sp
    x, y = sp.symbols('x, y')
    r = sp.sqrt(x**2 + y**2)
    s = sp.atan(y/x)
    sx = s.diff(x)
    sy = s.diff(y)
    rx = r.diff(x)
    ry = r.diff(y)
    omega = (x*sp.sqrt(x**2 + y**2) + y*(x**2 + y**2 -1))/(x*(x**2 + y**2-1)-y*sp.sqrt(x**2 + y**2))
    #omega = (y**3 + (x**2)*y - y- x)/(x*(y**2) + x**3 + y - x)
    w = omega.subs({x:r*sp.cos(s), y:r*sp.sin(s)})
    deriv = (rx + w*ry)/(sx + w*sy)
    print(sp.simplify(deriv))
