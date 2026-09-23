#Gradient Descent is an optimization algorithm used to find the minimum of a function. 


import numpy as np
import matplotlib.pyplot as plt

def y_fct(x):
    return x ** 2


def y_dericative(x):
    return 2 * x

x =np.arange(-100, 100, 0.1)
y = y_fct(x)

current_x = (50, y_fct(50))
learning_rate = 0.01 # learning rate is a hyperparameter that controls how much we adjust the weights of our model with respect to the loss gradient.
#how fast we want to move closer to the minimum point. A smaller learning rate means we take smaller steps towards the minimum, while a larger learning rate means we take larger steps.

for  _ in range(1000):
    new_x = current_x[0] - learning_rate * y_dericative(current_x[0])
    new_y = y_fct(new_x)
    current_x = (new_x, new_y)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y = x^2')
    plt.scatter(current_x[0], current_x[1], color='red')
    plt.pause(0.01)
    plt.clf()





#main 3D file :

def z_fct(x, y):
    return np.sin(5*x) * np.cos(5*y)/5

def calculate_gradient(x, y):
    dz_dx = 5 * np.cos(5*x) * np.cos(5*y)/5
    dz_dy = -5 * np.sin(5*x) * np.sin(5*y)/5
    return dz_dx, dz_dy

x = np.arange(-1, 1, 0.5)
y = np.arange(-1, 1, 0.5)
X, Y = np.meshgrid(x, y)
Z = z_fct(X, Y)


current_x = (0.7, 0.4 , z_fct(0.7, 0.4))
learning_rate = 0.01

ax = plt.subplot(projection='3d', computed_zorder=False)

for _ in range(1000):
    dz_dx, dz_dy = calculate_gradient(current_x[0], current_x[1])
    new_x = current_x[0] - learning_rate * dz_dx
    new_y = current_x[1] - learning_rate * dz_dy
    new_z = z_fct(new_x, new_y)
    current_x = (new_x, new_y, new_z)
    ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0)
    ax.scatter(current_x[0], current_x[1], current_x[2], color='red', zorder= 1)
    plt.pause(0.01)
    plt.clear()





