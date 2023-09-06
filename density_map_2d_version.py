import numpy as np
from scipy.spatial import cKDTree
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def density_function_2d(points, bandwidth, normalize=False):
    if normalize:
        points = [normalize_vector(point) for point in points]

    # Create a KDTree from the points for efficient nearest neighbor search
    kdtree = cKDTree(points)

    def density(lon, lat, kde=True):
        # Convert longitude and latitude to radians
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)

        # Convert longitude, latitude to Cartesian coordinates
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

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

def main():
    # Example usage
    points = np.random.randn(1000, 3)  # Example set of 1000 points on a sphere
    bandwidth = 0.1  # Bandwidth parameter for KDE

    # Generate a density function based on the points and bandwidth
    density_func = density_function_2d(points, bandwidth)

    # Define the map projection
    map = Basemap(projection='robin', lon_0=0, resolution='c')

    # Disable drawing country boundaries
    map.drawcountries(linewidth=0)

    # Generate a grid of longitude and latitude coordinates
    resolution = 100
    lon = np.linspace(-180, 180, resolution)
    lat = np.linspace(-90, 90, resolution)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Evaluate the density function on the grid of coordinates
    density_values = density_func(lon_grid.flatten(), lat_grid.flatten())

    # Reshape the density values to match the grid shape
    density_grid = density_values.reshape((resolution, resolution))

    # Convert longitude and latitude to map projection coordinates
    x, y = map(lon_grid, lat_grid)

    # Plot the density on the map
    plt.figure(figsize=(12, 6))
    map.contourf(x, y, density_grid, cmap='viridis')
    map.colorbar(label='Density')

    plt.title('Point Density on a Map')
    plt.show()

if __name__ == '__main__':
    main()