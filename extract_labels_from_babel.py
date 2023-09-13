import json
import numpy as np
from tqdm import tqdm


def read_json(jspath):
    jsfile = json.load(open(jspath, 'r'))
    motions = []
    labels = []
    categories = []
    for k in tqdm(jsfile.keys()):

        motions.append('/'.join(jsfile[k]['feat_p'].split('/')[-3:]))
        try:
            labels.append(jsfile[k]['seq_ann']['labels'][0]['proc_label'])
            categories.append(jsfile[k]['seq_ann']['labels'][0]['act_cat'])
        except KeyError:
            labels.append(jsfile[k]['seq_anns'][0]['labels'][0]['proc_label'])
            categories.append(jsfile[k]['seq_anns'][0]['labels'][0]['act_cat'])
    return motions, labels, categories

def construct_label_map_from_json(jspath_list):
    motions = []
    labels = []
    categories = []
    for j in jspath_list:
        x,y,z = read_json(j)
        motions += x
        labels += y
        categories += z
    
    babel_motions = {}
    babel_motions['motions'] = motions
    babel_motions['labels'] = labels
    babel_motions['catefories'] = categories

    np.savez_compressed('babel_motions.npz', **babel_motions)


def main():
    construct_label_map_from_json(['/mnt/workspace/liushuai_project/babel_v1.0_release/train.json',
                                    '/mnt/workspace/liushuai_project/babel_v1.0_release/val.json',
                                    '/mnt/workspace/liushuai_project/babel_v1.0_release/extra_val.json',
                                    '/mnt/workspace/liushuai_project/babel_v1.0_release/extra_train.json'])

if __name__ == '__main__':
    main()