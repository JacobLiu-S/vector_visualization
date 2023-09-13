import json
import os
from tqdm import tqdm
from glob import glob
import numpy as np

def read_motion_from_json(jspath):
    jsfile = json.load(open(jspath, 'r'))
    return '/'.join(jsfile['animation_file_path'].split('/')[-3:])


def iter_motion_json(root):
    motions = []
    place = [os.path.join(root,x) for x in ['BrooklynAlley',  'ConstructionSite',  'Downtown_West',  'Japanese',  'ModernCity',  'SF']]
    for p in tqdm(place):
        vlist = sorted(glob(p + '/*'))
        # import IPython; IPython.embed();exit()

        for v in tqdm(vlist, desc='video'):
            if v.endswith('nude'):
                continue
            if not os.path.exists(os.path.join(v, 'smpl_refit_withJoints_inCamSpace')):
                continue
            plist = sorted(glob(v + '/motions/*'))
            for per in plist:
                try:
                    jspath = os.path.join(per, 'meta.json')
                    motions.append(read_motion_from_json(jspath))
                except FileNotFoundError:
                    print(per)
                # motions.append(per.split('-')[-1])
    return motions

def get_labels_cat(motions):
    babel_motions = np.load('babel_motions.npz', allow_pickle=True)
    babel_motions_list = babel_motions['motions'].tolist()
    import IPython; IPython.embed(); exit()
    synbody_motions = {}
    smotions = [i.replace('stageii', 'poses') for i in motions]
    synbody_idx = []
    for m in smotions:
        synbody_idx.append(babel_motions_list.index(m))
    outlier = []
    for m in smotions:
        try:
            synbody_idx.append(babel_motions_list.index(m))
        except ValueError:
            outlier.append(m)
    synbody_motions['outlier'] = outlier
    synbody_motions['motion_idx'] = synbody_idx
    synbody_motions['cate'] = babel_motions['catefories'][synbody_idx]
    synbody_motions['labels'] = babel_motions['labels'][synbody_idx]
    np.savez('synbody_motions_sum.npz', **synbody_motions)


def main():
    get_labels_cat(iter_motion_json('/mnt/data/oss_beijing/pjlab-3090-openxrlab/share/Synbody_v1_1'))

if __name__ == '__main__':
    main()


