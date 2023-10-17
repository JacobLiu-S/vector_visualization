import numpy as np


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


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
    v1 = np.dot(vector, [0, 1, 0])
    v2 = np.dot(vector, [1, 0, 0])
    if v1 == 0:
        return 0 if v2 > 0 else -180
    if v2 == 0:
        return -90 if v1 > 0 else 90
    angle_y = np.degrees(np.arccos(np.dot(vector, [0, 1, 0])))
    angle_x = np.degrees(np.arccos(np.dot(vector, [1, 0, 0])))
    if 180 > angle_y > 90 and 90 > angle_x > 0:
        return angle_x
    if 180 > angle_y > 90 and 180 > angle_x > 90:
        # print(True)
        return angle_x
    if 90 > angle_y > 0 and 180 > angle_x > 90:
        # print(True)
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
    base_cam_rot = R.from_euler('zyx', [0, 0, 180], degrees=True).as_matrix()
    angles = []
    for v in tqdm(vectors):
        r = R.from_rotvec(v)
        rotmat = r.as_matrix()
        cam_in_body = np.linalg.inv(rotmat)

        # cam_x, cam_y, cam_z = cam_in_body[:, 0], cam_in_body[:, 1], cam_in_body[:, 2]
        # elevation_angle = calculate_elevation_angle(cam_z, [0,1,0])
        # angles.append(elevation_angle)

        cam_w_base = cam_in_body @ base_cam_rot
        x, y, z = R.from_matrix(cam_w_base).as_euler('xyz', degrees=True)
        angles.append(x)
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
        # angle = np.degrees(np.arccos(np.dot(cam_x, [0, 1, 0]))) - 90
        angle = calculate_z_rotation_angle(cam_x)
        angles.append(angle)
    return angles

def vis_follow_gta(vectors, option):
    if option == 1:
        return get_elevation_angle(vectors)
    elif option == 2:
        return get_azimuth_angles(vectors)
    elif option == 3:
        return get_z_rotation_angles(vectors)