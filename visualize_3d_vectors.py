import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib.colors import LogNorm

from density_calcuate import density_function, plot_density
from density_map_2d_version import density_function_2d


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def calculate_density(points, save=False):
    # Normalize the points to unit vectors
    normalized_points = [normalize_vector(point) for point in points]

    # Convert the normalized points to Cartesian coordinates
    x, y, z = zip(*normalized_points)

    # Perform kernel density estimation
    kde = gaussian_kde([x, y, z])

    # Evaluate the density at each point
    density = kde([x, y, z])

    # save the density
    if save:
        np.save('density.npy', density)

    return density

def rule_of_thumb_bandwidth(points):
    n = len(points)
    std_dev = np.std(points)
    iqr = np.percentile(points, 75) - np.percentile(points, 25)
    bandwidth = 0.9 * min(std_dev, iqr/1.34) * n**(-1/5)
    return bandwidth

def load_vectors(npzpath, samples=1000):
    if npzpath.endswith('.npy'):
        return np.load(npzpath).reshape(-1, 3)
    npfile = np.load(npzpath, allow_pickle=True)

    # for mmhuman3d human data
    smpl_global_orient = npfile['smpl'].item()['global_orient']

    sample_idx = np.random.choice(len(smpl_global_orient), samples, replace=False)

    return smpl_global_orient.reshape(-1, 3)[sample_idx]

def rotate_basis(vectors, basis=[0, 0, 1]):
    end_vectors = []
    for v in tqdm(vectors):
        r = R.from_rotvec(v)
        rotmat = r.as_matrix()
        end_vectors.append(rotmat @ basis)
    return np.vstack(end_vectors).reshape(-1, 3)

def main():
    parser = argparse.ArgumentParser(description='npz file path to be analyzed')
    parser.add_argument('--npz_path', help='the npz file to be analysed')
    parser.add_argument('--samples', default=10000, type=int, help='the number of samples to create the densities')
    args = parser.parse_args()
    vectors = rotate_basis(load_vectors(args.npz_path, samples=args.samples))
    # visualize_3d_vectors(vectors)

    # vectors = [[3, 4, 1], [1, 2, 2], [5, 5, 5]]  # List of vectors
    # densities = calculate_density(vectors)  # List of corresponding densities
    bandwidth = rule_of_thumb_bandwidth(vectors)
    print(f'Your bandwidth is {bandwidth}')
    density_func = density_function(vectors, bandwidth, True)
    
    # print(densities)
    resolution = 100
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    densities = density_func(x.flatten(), y.flatten(), z.flatten())
    print('densities calculated')
    plot_density(densities, resolution, x, y, z)

    # map density to 2d
    density_func_2d = density_function_2d(vectors, bandwidth)

    # Define the map projection
    map = Basemap(projection='robin', lon_0=0, resolution='c')

    # Disable drawing country boundaries
    # map.drawcountries(linewidth=0)

    # Generate a grid of longitude and latitude coordinates
    resolution = 100
    lon = np.linspace(-180, 180, resolution)
    lat = np.linspace(-90, 90, resolution)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Evaluate the density function on the grid of coordinates
    density_values_2d = density_func_2d(lon_grid.flatten(), lat_grid.flatten())

    # Reshape the density values to match the grid shape
    density_grid = density_values_2d.reshape((resolution, resolution))

    # Convert longitude and latitude to map projection coordinates
    x, y = map(lon_grid, lat_grid)

    # Plot the density on the map
    plt.figure(figsize=(12, 6))
    map.contourf(x, y, density_grid, cmap='viridis')
    map.colorbar(label='Density')

    plt.title('Point Density on a Map')
    # plt.show()
    density_2d_map_name = os.path.join('examples', os.path.basename(args.npz_path).split('.')[0] + '_density_2d.png')
    # plt.draw()
    plt.savefig(density_2d_map_name, dpi=100)

if __name__ == '__main__':
    main()


    # /mnt/workspace/liushuai_project/mmhuman3d/data/preprocessed_datasets/agora_train_smpl1.npz