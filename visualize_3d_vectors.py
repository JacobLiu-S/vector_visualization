import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
from collections import defaultdict
import math
from density_calcuate import density_function, plot_density, count_vectors_within_angle
from density_map_2d_version import density_function_2d
from sample_points_on_sphere import fibonacci_sphere, cartesian_to_spherical, spherical_to_latlon


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

    if samples > smpl_global_orient.shape[0]:
        return smpl_global_orient
    np.random.seed(0)
    sample_idx = np.random.choice(len(smpl_global_orient), samples, replace=False)

    # import IPython; IPython.embed();exit()
    return smpl_global_orient.reshape(-1, 3)[sample_idx]

def rotate_basis(vectors, basis=[0, 0, 1]):
    end_vectors = []
    for v in tqdm(vectors):
        r = R.from_rotvec(v)
        rotmat = r.as_matrix()
        end_vectors.append(rotmat @ basis)
    return np.vstack(end_vectors).reshape(-1, 3)

def calculate_elevation_angle(vector, plane_normal):
    # Normalize the vector and plane normal
    vector = np.array(vector) / np.linalg.norm(vector)
    plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
    
    # Calculate the dot product between the vector and plane normal
    dot_product = np.dot(vector, plane_normal)
    
    # Calculate the angle using the arccosine of the dot product
    angle = np.arccos(dot_product)
    
    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle)
    
    return angle_degrees - 90

def calculate_z_rotation_angle(vector):
    angle_y = np.degrees(np.arccos(np.dot(vector, [0, 1, 0])))
    angle_x = np.degrees(np.arccos(np.dot(vector, [1, 0, 0])))
    if 180 > angle_y > 90 and 90 > angle_x > 0:
        return angle_x
    if 180 > angle_y > 90 and 180 > angle_x > 90:
        print(True)
        return angle_x
    if 90 > angle_y > 0 and 180 > angle_x > 90:
        print(True)
        return - angle_y - 90
    if 90 > angle_y > 0 and 90 > angle_x > 0:
        return - angle_y
# def calculate_azimuth_angle(vector1, vector2):
#     angle = np.degrees(np.arccos(np.dot(vector1, vector2)))
#     if vector1[0] < 0:
#         angle = -angle
#     return angle
def angle_with_vertical(vector):
    vector2 = np.array([0, -1])  # 给定向量 [0, 1]
    # angle_rad = np.arctan2(np.cross(reference_vector, vector), np.dot(reference_vector, vector))
    # angle_deg = np.degrees(angle_rad)
    vector = normalize_vector(vector)
    # from left to right coord
    vector = [vector[0], -vector[1]]
    angle = np.degrees(np.arccos(np.dot(vector, vector2)))
    if vector[0] > 0:
        return -angle
    return angle


def calculate_density_count(angle_list):
    min_angle = math.floor(min(angle_list))
    max_angle = math.ceil(max(angle_list))
    # max_angle = max(int(max(angle_list)) + 1, 90)
    # angle_list_positive = [int(i + 90) for i in angle_list]
    angle_list_unique = [math.floor(x) for x in angle_list]
    # Divide the range between the minimum and maximum values into 1-degree intervals
    num_intervals = list(range(min_angle, max_angle))
    print(min_angle, max_angle)
    
    # Initialize the count for each interval to 0
    interval_counts = {}
    for  i in num_intervals:
        interval_counts[i] = 0
    for angle in angle_list:
        interval_counts[math.floor(angle)] += 1
    return interval_counts

def get_elevation_angle(vectors):
    angles = []
    for v in tqdm(vectors):
        r = R.from_rotvec(v)
        rotmat = r.as_matrix()
        cam_in_body = np.linalg.inv(rotmat)
        cam_x, cam_y, cam_z = cam_in_body[:, 0], cam_in_body[:, 1], cam_in_body[:, 2]
        elevation_angle = calculate_elevation_angle(cam_z, [0,1,0])
        angles.append(elevation_angle)
    return angles

def get_azimuth_angles(vectors):
    plt.figure(figsize=(12, 6))
    # cnt = 0
    angles = []
    for v in tqdm(vectors):
        r = R.from_rotvec(v)
        rotmat = r.as_matrix()
        cam_in_body = np.linalg.inv(rotmat)
        cam_x, cam_y, cam_z = cam_in_body[:, 0], cam_in_body[:, 1], cam_in_body[:, 2]
        cam_projection = cam_z[[0,2]]
        cam_projection = normalize_vector(cam_projection)
        # plt.plot([0, cam_projection[0]], [0, cam_projection[1]])
        angle = angle_with_vertical([-cam_projection[0], -cam_projection[1]])
        angles.append(angle)
        # if cnt % 50 == 0:
        #     plt.text(cam_projection[0], cam_projection[1], str(angle))
        # cnt += 1
    # densities = calculate_density_count(angles)
    # xx = sorted(densities.keys())
    # y = [densities[i] for i in xx]
    # plt.figure(figsize=(12, 6))
    # plt.plot(xx, y)
    # plt.savefig('debug_azimuth_angle.png')
    return angles

def get_z_rotation_angles(vectors):
    angles = []
    for v in tqdm(vectors):
        r = R.from_rotvec(v)
        rotmat = r.as_matrix()
        cam_in_body = np.linalg.inv(rotmat)
        cam_x, cam_y, cam_z = cam_in_body[:, 0], cam_in_body[:, 1], cam_in_body[:, 2]
        angle = np.degrees(np.arccos(np.dot(cam_x, [0, 1, 0]))) - 90
        angles.append(angle)
    return angles

def vis_follow_gta(vectors, option):
    if option == 1:
        return get_elevation_angle(vectors)
    elif option == 2:
        return get_azimuth_angles(vectors)
    elif option == 3:
        return get_z_rotation_angles(vectors)


def get_cam_in_per(vectors, option=1):
    """option = 1: elevation angle; 2: azimuth angle; 3: camara yz plane with z body rotation angle
    """
    base_cam_rot = R.from_euler('zyx', [0, 0, 180], degrees=True).as_matrix()
    per_in_map = R.from_matrix([[0,0,1], [1,0,0], [0,1,0]]).as_matrix()
    # rot_cam = R.from_euler('yxz', [y,x,0], degrees=True).as_matrix()
    output = []
    euler_z_angles = []
    elevation_angle_ = []
    for v in tqdm(vectors):
        r = R.from_rotvec(v)
        rotmat = r.as_matrix()
        cam_in_body = np.linalg.inv(rotmat)
        # print(cam_in_body)
        # import IPython; IPython.embed();exit()
        # exit()
        cam_x, cam_y, cam_z = cam_in_body[:, 0], cam_in_body[:, 1], cam_in_body[:, 2]
        # if option=2:
        # import IPython; IPython.embed();exit()
        if option == 1:
            elevation_angle = calculate_angle(cam_z, [0,1,0])
            elevation_angle_.append(elevation_angle)
        elif option == 2:
            # v1 = np.cross(cam_z, [0,1,0])
            cam_projection = cam_z[[0,2]]
            elevation_angle = angle_with_vertical([-cam_projection[0], -cam_projection[1]])
            # import IPython; IPython.embed();exit()
            elevation_angle_.append(elevation_angle)
        elif option == 3:
            elevation_angle = calculate_angle(cam_x, [0,1,0])
            elevation_angle_.append(elevation_angle)

        cam_w_base = cam_in_body @ base_cam_rot
        y, x, z = R.from_matrix(cam_w_base).as_euler('yxz', degrees=True)
        rot_cam = R.from_euler('yxz', [y,x,0], degrees=True).as_matrix()

        # cadi_coor = [-np.cos(x)*np.sin(y), np.sin(x), np.cos(y)*np.cos(x)]
        cadi_coor = (rot_cam @ [[0],[0],[1]]) # In this way, higher latitude, higherdensity.
        output.append((np.linalg.inv(per_in_map) @ cadi_coor).reshape(-1))
        # output.append(cadi_coor)
        euler_z_angles.append(z)
    return np.vstack(output).reshape(-1, 3), np.array(euler_z_angles), np.array(elevation_angle_)

def spherical_to_cartesian(longitude, latitude):
    x = math.cos(longitude) * math.cos(latitude)
    y = math.sin(longitude) * math.cos(latitude) 
    z = math.sin(latitude)
    
    return [x, y, z]

def plot_cam_pos(elevation_angles, azimuth_angles, out_path):
    # elevation_angles = vis_follow_gta(vectors, 1)
    # azimuth_angles = vis_follow_gta(vectors, 2)

    # per_in_map = R.from_matrix([[0,0,1], [1,0,0], [0,1,0]]).as_matrix()
    output = []
    for lat, lon in zip(elevation_angles, azimuth_angles):
        output.append([lon, lat])
    pos = np.vstack(output).reshape(-1, 2)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.scatter(pos[:, 0], pos[:, 1], s=1, c='navy', transform=ccrs.Geodetic())
    plt.title(f"Camera Position on a Map -- {os.path.basename(out_path)[:-4]}")
    plt.savefig(out_path, dpi=100)

def plot_cam_z(elevation_angles, azimuth_angles, z_angles, out_path):
    output = []
    for lat, lon, z in zip(elevation_angles, azimuth_angles, z_angles):
        output.append([lon, lat, z])
    pos = np.vstack(output).reshape(-1, 3)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.scatter(pos[:, 0], pos[:, 1], s=1, c=pos[:, 2], cmap='viridis', transform=ccrs.Geodetic())
    plt.colorbar(location='right', extend='both')
    plt.title(f"Camera Z Angles on a Map -- {os.path.basename(out_path)[:-4]}")
    plt.savefig(out_path, dpi=100)

def plot_cam_density(vectors, out_path):
    elevation_angles = vis_follow_gta(vectors, 1)
    azimuth_angles = vis_follow_gta(vectors, 2)

    per_in_map = R.from_matrix([[0,0,1], [1,0,0], [0,1,0]]).as_matrix()
    output = []
    for lat, lon in zip(elevation_angles, azimuth_angles):
        output.append([lon, lat])
    pos = np.vstack(output).reshape(-1, 2)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    resolution = 200
    num_points = resolution ** 2
    query_points = np.array(fibonacci_sphere(num_points))
    x, y, z = query_points[:,0], query_points[:,1], query_points[:,2]

    spherical_coordinates = np.array([cartesian_to_spherical(x, y, z) for x, y, z in query_points])
    latitudes_and_longitudes = np.array([spherical_to_latlon(r, theta, phi) for r, theta, phi in spherical_coordinates])

    densities, _ = count_vectors_within_angle(pos, np.column_stack((x.flatten(), y.flatten(), z.flatten())).reshape(resolution*resolution, 3), angle_threshold=1)
    
    plt.scatter(pos[:, 0], pos[:, 1], s=1, c='navy', transform=ccrs.Geodetic())

    plt.savefig(out_path, dpi=100)

def main():
    parser = argparse.ArgumentParser(description='npz file path to be analyzed')
    parser.add_argument('--npz_path', help='the npz file to be analysed')
    parser.add_argument('--samples', default=10000, type=int, help='the number of samples to create the densities')
    parser.add_argument('--kde', action='store_false', help='use KDE estimate to draw ')
    parser.add_argument('--cam_plot', action='store_true', help='use KDE estimate to draw ')
    args = parser.parse_args()
    vectors = load_vectors(args.npz_path, samples=args.samples)
    # get_azimuth_angles(vectors)
    # exit()
    # import IPython; IPython.embed(); exit()
    print(f'You sampled {vectors.shape[0]} samples')
    img_name = os.path.join('examples', os.path.basename(args.npz_path)[:-4]) + '_global_distributions.png'
    elevation_angles = vis_follow_gta(vectors, 1)
    azimuth_angles = vis_follow_gta(vectors, 2)
    z_angles = vis_follow_gta(vectors, 3)
    # plot_cam_pos(elevation_angles, azimuth_angles, img_name)
    img_name1 = os.path.join('examples', os.path.basename(args.npz_path)[:-4]) + '_global_rotations.png'
    plot_cam_z(elevation_angles, azimuth_angles, z_angles, img_name1)
    exit()
    for i in range(1, 4):
        # vectors, euler_z_angles, elevation_angles = get_cam_in_per(vectors, i)
        angles = vis_follow_gta(vectors, i)
        densities = calculate_density_count(angles)
        # import IPython; IPython.embed();exit()
        xx = sorted(densities.keys())
        y = [densities[i] for i in xx]
        plt.figure(figsize=(12, 6))
        plt.plot(xx, y)
        plt.xlabel('angle in degrees')
        plt.ylabel('# instances')
        img_prefix = os.path.join('examples', os.path.basename(args.npz_path).split('.')[0])
        if i == 1:
            plt.savefig(img_prefix + '_elevation_angle.png')
        elif i == 2:
            plt.savefig(img_prefix + '_azimuth_angle.png')
        elif i == 3:
            plt.savefig(img_prefix + '_z_rotation_angle.png')
    exit()
    # visualize_3d_vectors(vectors)

    # vectors = [[3, 4, 1], [1, 2, 2], [5, 5, 5]]  # List of vectors
    # densities = calculate_density(vectors)  # List of corresponding densities
    # bandwidth = rule_of_thumb_bandwidth(vectors)
    # print(f'Your bandwidth is {bandwidth}')
    # density_func = density_function(vectors, bandwidth, True)
    
    # print(densities)
    # resolution = 100
    # u = np.linspace(0, 2 * np.pi, resolution)
    # v = np.linspace(0, np.pi, resolution)
    # x = np.outer(np.cos(u), np.sin(v))
    # y = np.outer(np.sin(u), np.sin(v))
    # z = np.outer(np.ones_like(u), np.cos(v))

    resolution = 200
    num_points = resolution ** 2
    query_points = np.array(fibonacci_sphere(num_points))
    x, y, z = query_points[:,0], query_points[:,1], query_points[:,2]
    # if args.cam_plot:
    #     densities = z
    # densities = density_func(x.flatten(), y.flatten(), z.flatten(), args.kde)
    densities, _ = count_vectors_within_angle(vectors, np.column_stack((x.flatten(), y.flatten(), z.flatten())).reshape(resolution*resolution, 3), angle_threshold=1)
    print('densities calculated')
    print(densities.shape)
    # print(query_points.shape)
    plain_plot_density(query_points, np.log10(densities) )
    # plot_density(np.log10(densities), resolution, x, y, z)

    exit()
    # map density to 2d
    # density_func_2d = density_function_2d(vectors, bandwidth, True)

    # Define the map projection
    # map = Basemap(projection='robin', lon_0=0, resolution='c')
    # map = Basemap(projection='ortho', lat_0=0, lon_0=0)
    # Disable drawing country boundaries
    # map.drawcountries(linewidth=0)

    # Generate a grid of longitude and latitude coordinates
    resolution = 100
    lon = np.linspace(-180, 180, resolution)
    lat = np.linspace(-90, 90, resolution)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    lon_rad = np.radians(lon_grid)
    lat_rad = np.radians(lat_grid)
    # Convert longitude, latitude to Cartesian coordinates
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    # resolution = 100
    # lon = np.linspace(-180, 180, resolution)
    # lat = np.linspace(-90, 90, resolution)
    # lon_grid, lat_grid = np.meshgrid(lon, lat)

    # lon_rad = np.radians(lon_grid)
    # lat_rad = np.radians(lat_grid)
    # import IPython; IPython.embed(); exit()
    # # Convert longitude, latitude to Cartesian coordinates
    # x = np.cos(lat_rad) * np.cos(lon_rad)
    # y = np.cos(lat_rad) * np.sin(lon_rad)
    # z = np.sin(lat_rad)
    resolution = 100
    num_points = resolution ** 2
    query_points = np.array(fibonacci_sphere(num_points))
    # Concatenate x, y, and z coordinates into a single array
    query_points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

    # Evaluate the density function on the grid of coordinates
    # density_values_2d = density_func_2d(lon_grid.flatten(), lat_grid.flatten(), args.kde)
    density_values_2d, indices = count_vectors_within_angle(vectors, query_points.reshape(resolution*resolution, 3), 1)
    print('densities 2d calculated')
    # print(density_values.shape)
    # Reshape the density values to match the grid shape
    density_grid = density_values_2d.reshape((resolution, resolution)).reshape(-1)

    # Convert longitude and latitude to map projection coordinates
    spherical_coordinates = np.array([cartesian_to_spherical(x, y, z) for x, y, z in query_points])
    latitudes_and_longitudes = np.array([spherical_to_latlon(r, theta, phi) for r, theta, phi in spherical_coordinates])
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax = plt.axes(projection=ccrs.Mollweide())
    # ax = plt.axes(projection=ccrs.Robinson())
    
    # Mollweide
    plt.scatter(latitudes_and_longitudes[:, 1], latitudes_and_longitudes[:, 0], s=5, c=np.log10(density_grid), cmap='viridis', transform=ccrs.Geodetic())
    plt.colorbar(location='right', extend='both')

    # plt.savefig('test.png')
    # exit()
    # x, y = map(latitudes_and_longitudes[:, 1], latitudes_and_longitudes[:, 0])
    # lon_grid, lat_grid = latitudes_and_longitudes[:, 1].reshape(100, 100), latitudes_and_longitudes[:, 0].reshape(100, 100)
    # print(lon_grid.shape)
    # x, y = map(lon_grid, lat_grid)
    # x, y = map(query_points[:, 2], query_points[:, 1])

    # Plot the density on the map
    # plt.figure(figsize=(12, 6))
    # map.contourf(x, y, density_grid, cmap='viridis')
    # map.colorbar(label='Density')
    # colormap = map.pcolormesh(x, y, np.log10(density_grid), cmap='viridis')
    # colorbar = map.colorbar(colormap, location='right', pad='5%', extend='both')
    # map.scatter(x, y, s=5, c=density_grid, cmap='jet', marker='o')

    plt.title(f"Point Density on a Map -- {os.path.basename(args.npz_path).split('.')[0]}_cam_around_per_grid_cnt")
    # plt.show()
    density_2d_map_name = os.path.join('examples', os.path.basename(args.npz_path).split('.')[0] + f'_cam_around_per_grid_cnt_try1.png')
    # plt.draw()
    plt.savefig(density_2d_map_name, dpi=100)

    # exit()

    # plot z angles
    z_angles_min = []
    z_angles_max = []
    z_angles_abs = []
    for i in range(len(indices)):
        # import IPython; IPython.embed(); exit()
        try:
            min_z, max_z = min(euler_z_angles[indices[i]]), max(euler_z_angles[indices[i]])
            z_angles_min.append(min_z)
            z_angles_max.append(max_z)
            z_angles_abs.append(max(abs(euler_z_angles[indices[i]])))
        except ValueError:
            z_angles_min.append(0)
            z_angles_max.append(0)
            z_angles_abs.append(0)
    
    z_angles_min, z_angles_max, z_angles_abs = np.array(z_angles_min).reshape(resolution, resolution), np.array(z_angles_max).reshape(resolution, resolution), np.array(z_angles_abs).reshape(resolution, resolution)
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.scatter(latitudes_and_longitudes[:, 1], latitudes_and_longitudes[:, 0], s=5, c=z_angles_min, cmap='viridis', transform=ccrs.Geodetic())
    plt.colorbar(location='right', extend='both')
    # map = Basemap(projection='robin', lon_0=0, resolution='c')
    # colormap = map.pcolormesh(x, y, z_angles_min, cmap='viridis')
    # colorbar = map.colorbar(colormap, location='right', pad='5%', extend='both')
    plt.title(f"Point z_angles_min on a Map -- {os.path.basename(args.npz_path).split('.')[0]}_cam_around_per_z_angles_min")
    density_2d_map_name = os.path.join('examples', os.path.basename(args.npz_path).split('.')[0] + f'_cam_around_per_z_angles_min.png')
    plt.savefig(density_2d_map_name, dpi=100)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.scatter(latitudes_and_longitudes[:, 1], latitudes_and_longitudes[:, 0], s=5, c=z_angles_max, cmap='viridis', transform=ccrs.Geodetic())
    plt.colorbar(location='right', extend='both')
    # map = Basemap(projection='robin', lon_0=0, resolution='c')
    # colormap = map.pcolormesh(x, y, z_angles_max, cmap='viridis')
    # colorbar = map.colorbar(colormap, location='right', pad='5%', extend='both')
    plt.title(f"Point z_angles_max on a Map -- {os.path.basename(args.npz_path).split('.')[0]}_cam_around_per_z_angles_max")
    density_2d_map_name = os.path.join('examples', os.path.basename(args.npz_path).split('.')[0] + f'_cam_around_per_z_angles_max.png')
    plt.savefig(density_2d_map_name, dpi=100)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.scatter(latitudes_and_longitudes[:, 1], latitudes_and_longitudes[:, 0], s=5, c=z_angles_abs, cmap='viridis', transform=ccrs.Geodetic())
    plt.colorbar(location='right', extend='both')
    # map = Basemap(projection='robin', lon_0=0, resolution='c')
    # colormap = map.pcolormesh(x, y, z_angles_abs, cmap='viridis')
    # colorbar = map.colorbar(colormap, location='right', pad='5%', extend='both')
    plt.title(f"Point z_angles_abs on a Map -- {os.path.basename(args.npz_path).split('.')[0]}_cam_around_per_z_angles_abs")
    density_2d_map_name = os.path.join('examples', os.path.basename(args.npz_path).split('.')[0] + f'_cam_around_per_z_angles_abs.png')
    plt.savefig(density_2d_map_name, dpi=100)


if __name__ == '__main__':
    main()


    # /mnt/workspace/liushuai_project/mmhuman3d/data/preprocessed_datasets/agora_train_smpl1.npz