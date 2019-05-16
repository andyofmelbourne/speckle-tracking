import sympy
from pprint import pprint

"""
X,Y,p,sc_fs,c,zt,zx,zy,z = sympy.symbols('X,Y,p,sc_fs,c,zt,zx,zy,z', real=True)

eqs = [ sc_fs - (z+zx)/(z+zy) * zy / zx * N*uy / M*ux,
        c     - p*l*z* (z+zy)**2 / (zy*N*uy)**2,
        zt    - z + 0.5*(zx + zy)
      ]

eqs = [ sc_fs - (z+zx)/(z+zy) * zy / zx * Y / X,
        c     - p*z* (zt)**2 / (zy*Y)**2,
        zt    - z + zy
      ]
"""
a,b,c,zt,zx,zy,z,z1,z2,dz = sympy.symbols('a,b,c,zt,zx,zy,z,z1,z2,dz', real=True)

"""
eqs = [ z*(z/zx + 1) - a,
        z*(z/zy + 1) - c,
        z + (zx + zy)/2 - zt
      ]

sol = sympy.solve(eqs, [z, zx, zy], domain=sympy.Reals)
print(sol)
for i in range(2):
    print('z :\n', sol[i][0])
    print('\n\n')
    print('zx:\n', sol[i][1])
    print('\n\n')
    print('zy:\n', sol[i][2])
    print('\n\nOr:\n\n')
"""

eqs = [ (zt-z1)*(zt-dz)/(z1-dz) - a,
        (zt-z1)*(zt+dz)/(z1+dz) - b,
      ]

sol = sympy.solve(eqs, [z1, dz], domain=sympy.Reals)
print(sol)
for i in range(2):
    print('z1 :\n', sol[i][0])
    print('\n\n')
    print('dz\n', sol[i][1])
    print('\n\nOr:\n\n')
