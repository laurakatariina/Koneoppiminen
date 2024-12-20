import matplotlib.pyplot as plt
import numpy as np

points_x = []
points_y = []

def main():
    points()
    a, b = my_linfit(points_x, points_y)
    plt.plot(points_x, points_y, 'kx')
    xp = np.arange(-0, 20, 0.1)
    plt.plot(xp, a * xp + b, 'r-')
    print(f'My fit: a={a} and b={b}')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Fitted Line')
    plt.show()

# Linear fit
def my_linfit(x, y):
    # List to array
    x = np.array(x)
    y = np.array(y)

    # Define numbers
    N = len(x)
    xy = np.sum(x * y)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    sq_sum = np.sum(x**2)

    # Based on derivative calculations
    a = (N * xy - x_sum * y_sum) / (N * sq_sum - x_sum**2)
    b = (y_sum - a * x_sum) / N

    return a, b

# Collecting points
def points():
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', onclick)

    ax.set_xlim([-1, 20])
    ax.set_ylim([-1, 20])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Select points with left mouse click')
    plt.show()

# Mouse functions
def onclick(event):
    if event.button == 1:
        points_x.append(event.xdata)
        points_y.append(event.ydata)
        plt.plot(event.xdata, event.ydata, 'ro')
        plt.draw()
    elif event.button == 3:
        plt.close()

main()

