'''
Este script integra la ecuacion de movimiento usando el metodo
de Runge Kutta orden 3 implementado en una version propia.
Se utilizan las condiciones iniciales y=4.0, v=0 y se grafica
la trayectoria en el espacio.

'''

import numpy as np
import matplotlib.pyplot as plt

MU=1.762

plt.figure(1)
plt.clf()


def f(y, v, mu=MU):
    return v, -y-mu*((y**2)-1)*v

def get_k1(y_n, v_n, h, f):
    f_eval = f(y_n, v_n)
    return h * f_eval[0], h * f_eval[1]

def get_k2(y_n, v_n, h, f):
    k1 = get_k1(y_n, v_n, h, f)
    f_eval = f(y_n + k1[0]/2., v_n + k1[1]/2.)
    return h * f_eval[0], h * f_eval[1]

def get_k3(y_n, v_n, h, f):
    k1 = get_k1(y_n, v_n, h, f)
    k2 = get_k2(y_n, v_n, h, f)
    f_eval= f(y_n - k1[0] - 2*k2[0], v_n - k1[1] - 2*k2[1])
    return h * f_eval[0], h * f_eval[1]

def rk3_step(y_n, v_n, h, f):
    k1 = get_k1(y_n, v_n, h, f)
    k2 = get_k2(y_n, v_n, h, f)
    k3 = get_k2(y_n, v_n, h, f)
    y_n1 = y_n + (1/6.) * (k1[0] + 4*k2[0] + k3[0])
    v_n1 = v_n + (1/6.) * (k1[1] + 4*k2[1] + k3[1])
    return y_n1, v_n1

N_steps = 40000
h = 20. * np.pi / N_steps
y = np.zeros(N_steps)
v = np.zeros(N_steps)

y[0] = 4.0
v[0] = 0
for i in range(1, N_steps):
    rk3 = rk3_step(y[i-1], v[i-1], h, f)
    y[i] = rk3[0]
    v[i] = rk3[1]

#Plot de y vs dy/ds
plt.plot(y, v, 'g')
plt.draw()
plt.title('Grafico de la trayectoria en el espacio. Condiciones iniciales: y=4, v=0')
plt.xlabel('y(s)', fontsize=16)
plt.ylabel('dy(s)/ds', fontsize=16)
plt.savefig('figura2.png')

'''
#Plot de y vs s. Para graficar sacar las comillas arriba y abajo.
s_values = np.linspace(0, 20. * np.pi, N_steps)
plt.plot(s_values, y)
plt.draw()
plt.title('Grafico de y(s) versus s. Condiciones iniciales: y=4, v=0')
plt.xlabel('s', fontsize=16)
plt.ylabel('y(s)', fontsize=16)
plt.savefig('figura4.png')
'''

plt.show()
