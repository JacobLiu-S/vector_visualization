import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

def find_closest_vectors(small_array, big_array, vector):
    small_rotations = np.apply_along_axis(rotation_matrix_from_vector, 1, small_array)
    big_rotations = np.apply_along_axis(rotation_matrix_from_vector, 1, big_array)

    small_vectors = np.matmul(small_rotations, vector)
    big_vectors = np.matmul(big_rotations, vector)

    tree = KDTree(big_vectors)
    distances, closest_indices = tree.query(small_vectors, k=1)
    # print(closest_indices)
    # print(distances)
    # for i in range(5):
    #     print(angle_between_vectors(small_vectors[i], big_vectors[closest_indices][i]))
    return closest_indices

def rotation_matrix_from_vector(rotation_vector):
    r = R.from_rotvec(rotation_vector)
    rotmat = r.as_matrix()
    return rotmat

def angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)

# example
def main():
    small_array = np.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                        [1.0, 1.1, 1.2],
                        [1.3, 1.4, 1.5]])

    big_array = np.random.randn(10000, 3)

    vector = [0.0, 0.0, 1.0]

    closest_idx = find_closest_vectors(small_array, big_array, vector)
    print(closest_idx)

if __name__ == '__main__':
    main()
