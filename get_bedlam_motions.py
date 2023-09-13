import os
import numpy as np
from glob import glob
from tqdm import tqdm


def read_motion_from_npz(nppath):
    npfile = np.load(nppath)
    motion = npfile['motion_info'][2]
    # motion_list = '/'.join(motion.split('/')[-3:])
    # motion_list[-1] = '_'.join(motion_list[-1].split('_')[:2])
    # motion = '/'.join(motion_list)
    return '/'.join(motion.split('/')[-3:])

def main():
    root = '/mnt/data/oss_beijing/pjlab-3090-openxrlab/share/neutral_ground_truth_motioninfo'
    babel_motions = np.load('babel_motions.npz', allow_pickle=True)
    babel_motions_list = babel_motions['motions'].tolist()

    plist = sorted(glob(os.path.join(root, '*')))
    bedlam_idx = []
    bedlam_motions = []
    bmotions = []
    for p in tqdm(plist):
        s = glob(os.path.join(p, '*'))
        for ss in tqdm(s, desc='person'):
            npzlist = glob(ss + '/*.npz')
            for nppath in npzlist:
                motion = read_motion_from_npz(nppath)
                bmotions.append(motion)
                # tmp_idx = []
                # tmp_motions = []
                # for i in range(len(babel_motions_list)):
                #     if motion in babel_motions_list[i]:
                #         tmp_idx.append(i)
                #         tmp_motions.append(babel_motions_list[i])
                # bedlam_idx.append(tmp_idx)
                # bedlam_motions.append(tmp_motions)
    import IPython; IPython.embed();exit()

if __name__ == '__main__':
    main()
