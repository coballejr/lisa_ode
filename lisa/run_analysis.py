from .prolongation import Prolongation
from .lie_field import X1, X2, X3, X4, X5, X6

# Convection-diffusion equation
# vu_x - ku_xx = 0

needed_prolongations = ['x', 'xx']

print('X1:')
P1 = Prolongation(X1())
for partial in needed_prolongations:
    print(partial, P1.prolong(partial))
print('==================================')

print('X2:')
P2 = Prolongation(X2())
for partial in needed_prolongations:
    print(partial, P2.prolong(partial))
print('==================================')

print('X3:')
P3 = Prolongation(X3())
for partial in needed_prolongations:
    print(partial, P3.prolong(partial))
print('==================================')

print('X4:')
P4 = Prolongation(X4())
for partial in needed_prolongations:
    print(partial, P4.prolong(partial))
print('==================================')

print('X5:')
P5 = Prolongation(X5())
for partial in needed_prolongations:
    print(partial, P5.prolong(partial))
print('==================================')

print('X6:')
P6 = Prolongation(X6())
for partial in needed_prolongations:
    print(partial, P6.prolong(partial))




