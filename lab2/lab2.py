import numpy as nmp
import matplotlib.pyplot as pyplt
from scipy.interpolate import CubicSpline

def draw(x, y):
    spline = CubicSpline(x, y)
    x_val = nmp.linspace(0.0, 1.0, 100)
    pyplt.plot(x_val, f(x_val), 'g', label='function')
    pyplt.plot(x_val, spline(x_val), 'r--', label='spline')
    pyplt.legend()
    pyplt.grid(True)
    pyplt.title('n = {}'.format(n))
    #pyplt.legend()
    pyplt.show()
	
def f(x):
    return nmp.exp(x)*x + x**2

for n in [3, 10]:
    x_arr = nmp.linspace(0.0, 1.0, n)
    y_arr = f(x_arr)
    draw(x_arr, y_arr)
