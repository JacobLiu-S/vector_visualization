import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fibonacci_sphere(samples=1):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # 黄金角度
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y坐标
        radius = np.sqrt(1 - y*y)  # 半径

        theta = phi * i  # 角度

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return points

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_latlon(r, theta, phi):
    lat = np.rad2deg(np.pi/2 - theta)
    lon = np.rad2deg(phi)
    return lat, lon

def main():
    num_points = 10000
    points = fibonacci_sphere(num_points)

    # 绘制结果
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points])
    plt.show()

if __name__ == '__main__':
    main()
