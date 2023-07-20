import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz

# NTU RGB+D Skeleton 120 Configurations: https://arxiv.org/pdf/1905.04757.pdf
training_subjects = set([
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
])

training_setups = set(range(2, 33, 2))
#training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(file_list,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    ignored_samples = []
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]

    sample_name = []
    sample_label = []
    sample_paths = []
    for folder, filename in sorted(file_list):
        if filename in ignored_samples:
            continue

        path = os.path.join(folder, filename)
        setup_loc = filename.find('S')
        subject_loc = filename.find('P')
        action_loc = filename.find('A')
        setup_id = int(filename[(setup_loc+1):(setup_loc+4)])
        subject_id = int(filename[(subject_loc+1):(subject_loc+4)])
        action_class = int(filename[(action_loc+1):(action_loc+4)])

        if benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xset':
            istraining = (setup_id in training_setups)
        else:
            raise ValueError(f'Unsupported benchmark: {benchmark}')

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError(f'Unsupported dataset part: {part}')

        if issample:
            sample_paths.append(path)
            sample_label.append(action_class - 1)   # to 0-indexed


    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--part1-path', default='/home/data/skeleton/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--part2-path', default='/home/data/skeleton/nturgbd_raw/nturgb+d_skeletons120/')
    #parser.add_argument(
    #    '--data_path', default='/home/data/skeleton/nturgbd_raw/nturgb+d_skeletons')
    parser.add_argument(
        '--ignored_sample_path',
        default='/home/data/skeleton/nturgbd_raw/NTU_RGBD120_samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/NTU120')

    benchmark = ['xsub', 'xset']
    part = ['train', 'val']
    arg = parser.parse_args()

    # Combine skeleton file paths
    file_list = []
    for folder in [arg.part1_path, arg.part2_path]:
        for path in os.listdir(folder):
            file_list.append((folder, path))

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(file_list, out_path, arg.ignored_sample_path, benchmark=b, part=p)
