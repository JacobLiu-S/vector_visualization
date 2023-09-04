import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib.colors import LogNorm

from density_calcuate import density_function, plot_density


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

def plot_normalized_vectors(vectors, densities):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # fig, ax = plt.subplots(figsize = (12, 7), projection='3d')

    normalized_vectors = [normalize_vector(vector) for vector in vectors]

    for vector, density in tqdm(zip(normalized_vectors, densities)):
        ax.scatter(vector[0], vector[1], vector[2], c=density, cmap='viridis', norm=LogNorm())

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Plot x-axis arrow
    ax.quiver(-1, 0, 0, 1, 0, 0, color='k', arrow_length_ratio=0.1, length=2)
    # Plot y-axis arrow
    ax.quiver(0, -1, 0, 0, 1, 0, color='k', arrow_length_ratio=0.1, length=2)
    # Plot z-axis arrow
    ax.quiver(0, 0, -1, 0, 0, 1, color='k', arrow_length_ratio=0.1, length=2)
    
    ax.axis('on')
    plt.show()
    plt.savefig('examples/normalized_vectors.png')


def load_vectors(npzpath, samples=1000):
    npfile = np.load(npzpath, allow_pickle=True)

    # for mmhuman3d human data
    smpl_global_orient = npfile['smpl'].item()['global_orient']

    sample_idx = np.random.choice(len(smpl_global_orient), samples)

    return smpl_global_orient.reshape(-1, 3)[sample_idx]

def rotate_basis(vectors, basis=[0, 0, 1]):
    end_vectors = []
    for v in tqdm(vectors):
        r = R.from_rotvec(v)
        rotmat = r.as_matrix()
        end_vectors.append(rotmat @ basis)
    return np.vstack(end_vectors).reshape(-1, 3)

def sphere_to_rectangular(sphere_points, densities, resolution):
    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(sphere_points[:, 0]) * np.cos(sphere_points[:, 1])
    y = np.sin(sphere_points[:, 0]) * np.sin(sphere_points[:, 1])
    z = np.cos(sphere_points[:, 0])

    # Convert Cartesian coordinates to latitude and longitude
    lat = np.arcsin(z)
    lon = np.arctan2(y, x)

    # Normalize longitude values to range [-pi, pi]
    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    # Convert longitude and latitude to equirectangular projection
    x_rect = (lon + np.pi) / (2 * np.pi) * resolution
    y_rect = (lat + np.pi / 2) / np.pi * resolution

    # Create a rectangular grid for the density map
    density_map = np.zeros((resolution, resolution))

    # Populate the density map with the corresponding densities
    for i in range(len(densities)):
        x_idx = int(x_rect[i])
        y_idx = int(y_rect[i])
        density_map[y_idx, x_idx] += densities[i]

    return density_map

def main():
    parser = argparse.ArgumentParser(description='npz file path to be analyzed')
    parser.add_argument('--npz_path', help='the npz file to be analysed')
    args = parser.parse_args()
    vectors = rotate_basis(load_vectors(args.npz_path))
    # visualize_3d_vectors(vectors)

    # vectors = [[3, 4, 1], [1, 2, 2], [5, 5, 5]]  # List of vectors
    # densities = calculate_density(vectors)  # List of corresponding densities
    bandwidth = 0.1
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
    # plot_density(densities, resolution, x, y, z)

    # plot_normalized_vectors(vectors, densities)

    # sphere_points = np.array(vectors)
    # print(x.shape)
    sphere_points = np.zeros((10000, 3))
    sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2] = x.flatten(), y.flatten(), z.flatten()
    print(densities.shape)
    density_map = sphere_to_rectangular(sphere_points, densities, resolution)

    # Plot the rectangular density map
    plt.imshow(density_map, cmap='viridis')
    plt.colorbar()
    # plt.savefig('examples/density_map.png')
    plt.show()

if __name__ == '__main__':
    main()


    # /mnt/workspace/liushuai_project/mmhuman3d/data/preprocessed_datasets/agora_train_smpl1.npz