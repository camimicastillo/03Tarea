'''
Este script resuelve numericamente el set de ecuaciones del sistema
de Lorenz, mediante el uso del metodo predeterminado de Runge Kutta orden
4 que corresponde al comando ODE.

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

#Parametros
sigma = 10
beta = 8/3
rho = 28

#Condiciones iniciales
x0 = 5
y0 = 5
z0 = 5
v0 = [x0, y0, z0]
t0 = 0

def f_to_solve(t, v, S=sigma, B=beta, R=rho):
    x, y, z = v
    return [S*(y-x), x*(R-z)-y, x*y-B*z]

#Creamos el 'resolvedor'
r = ode(f_to_solve)
r.set_integrator('dopri5')
r.set_initial_value(v0, t0)

#Guardamos las variables a medida que progresamos
t_values = np.linspace(t0, 100, 5000)
x_values = np.zeros(len(t_values))
y_values = np.zeros(len(t_values))
z_values = np.zeros(len(t_values))

for i in range(len(t_values)):
    r.integrate(t_values[i])
    x_values[i], y_values[i], z_values[i] = r.y


fig = plt.figure(1)
fig.clf()

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')


ax.plot(x_values, y_values, z_values)

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')

ax.set_title('Grafico del Atractor de Lorenz')

plt.savefig('figura5.png')

plt.draw()
plt.show()
