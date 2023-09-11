import numpy as np
import subprocess
import argparse
from find_closest_go import find_closest_vectors
import os
import cv2

DATASETS = {
    'agora': 'agora',
    'synbody': 'Synbody_v1_1',
    'h36m':'h36m',
    '3dpw':'pw3d'
}

BEDLAM_ROOT = '/mnt/data/oss_beijing/pjlab-3090-openxrlab/share/mmhuman3d_datasets/datasets/bedlam/training_images'

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def draw_bbox(image_path, bbox):
    # Read the image
    try:
        assert os.path.exists(image_path)
    except:
        print(image_path)
        exit()
    image = cv2.imread(image_path)

    # Extract the bounding box coordinates and dimensions
    if len(bbox) == 5:
        x, y, w, h, _ = bbox
    else:
        x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Draw the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the image with the bounding box
    # cv2.imwrite(outpath, image)
    return image

def save_horizontal_concatenation(image1, image2, output_path):
    # Get the dimensions of the input images
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Calculate the maximum height and width
    max_height = max(height1, height2)
    max_width = max(width1, width2)

    # Calculate the required padding for each image
    pad_height1 = max_height - height1
    pad_width1 = max_width - width1
    pad_height2 = max_height - height2
    pad_width2 = max_width - width2

    # Pad the images with zeros
    padded_image1 = cv2.copyMakeBorder(image1, 0, pad_height1, 0, pad_width1, cv2.BORDER_CONSTANT, value=0)
    padded_image2 = cv2.copyMakeBorder(image2, 0, pad_height2, 0, pad_width2, cv2.BORDER_CONSTANT, value=0)

    # Concatenate images horizontally
    concatenated_image = np.hstack((padded_image1, padded_image2))

    # Save the concatenated image
    cv2.imwrite(output_path, concatenated_image)

def main():
    parser = argparse.ArgumentParser(description='npz file path to be analyzed')
    # parser.add_argument('--npz_src', help='the npz file to be searched')
    # parser.add_argument('--src_name', help='in which the source npz represents')
    # parser.add_argument('--npz_dst', help='the destination npz file')
    # parser.add_argument('--dst_name', help='in which the destination npz represents')
    args = parser.parse_args()

    # for bedlam format
    npfile = np.load('bedlam_global_cam.npz')
    samples = 3
    np.random.seed(0)
    idx = np.random.choice(len(npfile['image_path']), 3, replace=False)
    print(npfile['image_path'][idx])
    small_vec = npfile['global_orient'][idx]

    samples_src = 1000
    idx_src_sample = np.random.choice(len(npfile['image_path']), 1000, replace=False)
    big_vec = npfile['global_orient'][idx_src_sample]

    vector = [0, 0, 1]
    vector = normalize_vector(vector)
    closest_idx = find_closest_vectors(small_vec, big_vec, vector)
    print(closest_idx)
    print(npfile['image_path'][idx_src_sample][closest_idx])

    for i in range(samples):
        # print(os.path.join(BEDLAM_ROOT, npfile['image_path'][idx[i]]))
        im1 = draw_bbox(os.path.join(BEDLAM_ROOT, npfile['image_path'][idx[i]]), npfile['bbox'][idx[i]])
        im2 = draw_bbox(os.path.join(BEDLAM_ROOT, npfile['image_path'][idx_src_sample][closest_idx[i]]), npfile['bbox'][idx_src_sample][closest_idx[i]])
        output_path = os.path.join('imgs', npfile['image_path'][idx[i]].split('/')[-1][:-4] +'---'+ npfile['image_path'][idx_src_sample][closest_idx[i]].split('/')[-1][:-4] + '.png')
        save_horizontal_concatenation(im1, im2, output_path)



    # for humandata
    # np_dst = np.load(args.npz_dst, allow_pickle=True)
    # samples = 3
    # np.random.seed(0)
    # idx = np.random.choice(len(np_dst['image_path']), 3, replace=False)
    # print(np_dst['image_path'][idx])
    # small_vec = np_dst['smpl'].item()['global_orient'][idx]


    # np_src = np.load(args.npz_src, allow_pickle=True)
    # samples_src = 10000
    # idx_src_sample = np.random.choice(len(np_src['image_path']), 10000, replace=False)
    # big_vec = np_src['smpl'].item()['global_orient'][idx_src_sample]

    # vector = [0, 0, 1]
    # vector = normalize_vector(vector)
    # closest_idx = find_closest_vectors(small_vec, big_vec, vector)
    # print(closest_idx)
    # print(np_src['image_path'][closest_idx])

    # for i in range(samples):
    #     cmd = ['cp', os.path.join('/mnt/workspace/liushuai_project/mmhuman3d/data/datasets', DATASETS[args.dst_name], np_dst['image_path'][idx[i]]), 'imgs']
    #     subprocess.call(cmd)
    #     cmd = ['cp', os.path.join('/mnt/workspace/liushuai_project/mmhuman3d/data/datasets', DATASETS[args.src_name], np_src['image_path'][closest_idx[i]]), 'imgs']
    #     subprocess.call(cmd)

    # for i in range(samples):
        # im1 = draw_bbox(os.path.join('imgs', np_dst['image_path'][idx[i]].split('/')[-1]), np_dst['bbox_xywh'][idx[i]])
        # im2 = draw_bbox(os.path.join('imgs', np_src['image_path'][closest_idx[i]].split('/')[-1]), np_src['bbox_xywh'][closest_idx[i]])
        # output_path = os.path.join('imgs', np_dst['image_path'][idx[i]].split('/')[-1][:-4] +'---'+ np_src['image_path'][closest_idx[i]].split('/')[-1][:-4] + '.png')
        # save_horizontal_concatenation(im1, im2, output_path)

if __name__ == '__main__':
    main()
    
