import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

# def count_vectors_within_angle(vectors, target_vectors, angle_threshold):
#     # Normalize the vectors
#     # vectors_unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
#     # target_vectors_unit = target_vectors / np.linalg.norm(target_vectors, axis=1, keepdims=True)
#     vectors_unit = vectors
#     target_vectors_unit = target_vectors
    
#     # Calculate the angles (in radians) between the vectors
#     angles = np.arccos(np.dot(vectors_unit, target_vectors_unit.T))
    
#     # Count the number of vectors that have an angle smaller than the given threshold
#     counts = np.sum(angles < np.deg2rad(angle_threshold), axis=0)
    
#     return counts

def count_vectors_within_angle(vectors, target_vectors, angle_threshold):
    # Normalize the vectors
    # vectors_unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # target_vectors_unit = target_vectors / np.linalg.norm(target_vectors, axis=1, keepdims=True)
    vectors_unit = vectors
    target_vectors_unit = target_vectors
    
    # Calculate the angles (in radians) between the vectors
    angles = np.arccos(np.dot(vectors_unit, target_vectors_unit.T))
    
    # Count the number of vectors that have an angle smaller than the given threshold
    counts = np.sum(angles < np.deg2rad(angle_threshold), axis=0)

    # Find the indices of vectors that have an angle smaller than the given threshold
    a = angles < np.deg2rad(angle_threshold)
    indices = []
    for x in range(len(target_vectors)):
        indices.append(np.where(a[:, x] == True)[0])
    
    # import IPython; IPython.embed()
    return counts, indices


def density_function(points, bandwidth, normalize=False):
    if normalize:
        points = [normalize_vector(point) for point in points]

    # Create a KDTree from the points for efficient nearest neighbor search
    kdtree = cKDTree(points)

    def density(x, y, z, kde=True):
        # Concatenate x, y, and z coordinates into a single array
        query_points = np.column_stack((x, y, z))

        # Find the distances and indices of the nearest neighbors
        distances, _ = kdtree.query(query_points, k=1)

        # Calculate the kernel density estimate based on the distances
        if kde:
            density_values = 1 / (bandwidth * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (distances / bandwidth) ** 2)
            return density_values
            
        return distances

    return density


def plot_density(density_values, resolution, x, y, z):
    density_grid = density_values.reshape((resolution, resolution))
    # Plot the density on the sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(density_grid), rstride=1, cstride=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # # Plot x-axis arrow
    # ax.quiver(-2, 0, 0, 2, 0, 0, color='k', arrow_length_ratio=0.1, length=4)
    # # Plot y-axis arrow
    # ax.quiver(0, -2, 0, 0, 2, 0, color='k', arrow_length_ratio=0.1, length=4)
    # # Plot z-axis arrow
    # ax.quiver(0, 0, -2, 0, 0, 2, color='k', arrow_length_ratio=0.1, length=4)
    ax.set_title('Point Density on a Sphere')

    # Create a ScalarMappable to map density values to colors
    sm = ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(density_values)
    fig.colorbar(sm, label='Density')

    plt.show()

def main():
# Example usage
    points = np.random.randn(1000, 3)  # Example set of 1000 points on a sphere
    bandwidth = 0.1  # Bandwidth parameter for KDE

    # Generate a density function based on the points and bandwidth
    density_func = density_function(points, bandwidth)

    # Generate a grid of points on a sphere
    resolution = 100
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Evaluate the density function on the grid of points
    density_values = density_func(x.flatten(), y.flatten(), z.flatten())

    plot_density(density_values, resolution, x, y, z)

if __name__ == '__main__':
    main()