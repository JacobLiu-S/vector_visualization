import numpy as np
import matplotlib.pyplot as plt
import argparse

def analyze_motion_labels(motion_dict):
    labels = motion_dict['labels']
    unique_labels = list(set(labels))
    labels_dict = {}
    for l in unique_labels:
        labels_dict[l] = 0
    for l in labels:
        labels_dict[l] += 1
    sorted_dict = dict(sorted(labels_dict.items(), key=lambda x: x[1]))
    return sorted_dict


def draw_labels(sorted_dict, dataset):
    keys = sorted_dict.keys()
    values = sorted_dict.values()
    w = len(keys) // 10
    plt.figure(figsize=(w, 6))
    bar_plot = plt.bar(keys, values, width=0.8)

    for i, rect in enumerate(bar_plot):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height),
                ha='center', va='bottom', rotation=90, fontsize=6)

    plt.xlabel('motions')
    plt.ylabel('Values')
    plt.title('Bar Chart')

    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    prefix = dataset.split('_')[0]
    plt.savefig('imgs/' + prefix+'_motion_labels.png')

def draw_both_bar(dict1, dict2, draw_split=False):
    merged_dict = {}

    keys = set(list(dict1.keys()) + list(dict2.keys()))

    for key in keys:
        value1 = dict1.get(key, 0)
        value2 = dict2.get(key, 0)
        merged_dict[key] = (value1, value2)

    merged_dict = dict(sorted(merged_dict.items(), key=lambda x: sum(x[1])))
    keys = merged_dict.keys()
    values1, values2 = zip(*merged_dict.values())

    if not draw_split:
        w = len(keys) // 10
        fig, ax = plt.subplots(figsize=(w, 20))

        x = np.arange(len(keys))
        # Plot the first segment of the bars representing values from dict1
        ax.bar(x, values1, label='bedlam')

        # Plot the second segment of the bars representing values from dict2
        ax.bar(x, values2, bottom=values1, label='synbody')


        plt.xlabel('Keys')
        plt.ylabel('Values')
        plt.title('Combined Bar Chart')
        plt.legend()

        plt.xticks(ticks=x, labels=keys, rotation=90, ha='center', fontsize=6)
        plt.yticks(fontsize=6)

        plt.savefig('imgs/bedlam_synbody_motion_labels.png')
    else:
        n = len(keys)
        split = n // 3
        keys_split = [list(keys)[i:i + split] for i in range(0, n, split)]
        values1_split = [list(values1)[i:i + split] for i in range(0, n, split)]
        values2_split = [list(values2)[i:i + split] for i in range(0, n, split)]

        # fig, axs = plt.subplots(1, 3, figsize=(n//10, 20), sharey=True)
        for i in range(3):
            # fig, ax = plt.subplots(1, 3, figsize=(n // 30, 20))
            fig, ax = plt.subplots(figsize=(n//10, 20))
            # ax = axs[i]
            ax.bar(range(len(keys_split[i])), values1_split[i], label='bedlam')
            ax.bar(range(len(keys_split[i])), values2_split[i], bottom=values1_split[i], label='synbody')

            ax.set_xlabel('Keys')
            ax.set_ylabel('Values')
            ax.set_title(f'Subplot {i+1}')

            ax.set_xticks(range(len(keys_split[i])))
            ax.set_xticklabels(keys_split[i], rotation=90, ha='center', fontsize=10)
            ax.legend()
            plt.savefig(f'imgs/bedlam_synbody_motion_labels_split_{i}.png')


def main():
    parser = argparse.ArgumentParser(description='npz file path to be analyzed')
    parser.add_argument('--npz_path', help='the npz file to be analysed')
    parser.add_argument('--npz_path2', default=None, help='the npz file to be analysed')
    parser.add_argument('--draw_split', action='store_true', help='the npz file to be analysed')
    args = parser.parse_args()
    if args.npz_path2:
        dict1 = analyze_motion_labels(np.load(args.npz_path))
        dict2 = analyze_motion_labels(np.load(args.npz_path2))
        draw_both_bar(dict1, dict2, args.draw_split)
    else:
        npfile = np.load(args.npz_path)
        sorted_dict = analyze_motion_labels(npfile)
        draw_labels(sorted_dict, args.npz_path)


if __name__ == '__main__':
    main()