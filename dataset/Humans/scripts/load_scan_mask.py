# -*- coding: utf-8 -*-
# Script to parse ground-truth masks for a scan sequence
# Copyright (c) [2017] [MPI for Intelligent Systems]

from argparse import ArgumentParser
from os.path import exists, join
import h5py


def parse_masks(path):
    with h5py.File(path, 'r') as f:
        seq_frames = sorted(f.keys())
        n_frames = len(seq_frames)
        n_verts = 0
        n_gt_verts = 0
        for frame in seq_frames:
            mask = f[frame][:]
            n_verts += mask.size
            n_gt_verts += mask.sum()
        perc = 100 * (float(n_gt_verts) / n_verts)
        print(
            '%d out of %d vertices (%.2f%%) in the sequence are accurately registered'
            % (n_gt_verts, n_verts, perc))


if __name__ == '__main__':

    # Subject ids
    sids = [
        '50002', '50004', '50007', '50009', '50020', '50021', '50022', '50025',
        '50026', '50027'
    ]
    # Sequences available for each subject id are listed in scripts/subjects_and_sequences.txt

    parser = ArgumentParser(
        description='Parse ground-truth masks for scan sequence')
    parser.add_argument('--path',
                        type=str,
                        default='../masks',
                        help='folder containing mask files in hdf5 format')
    parser.add_argument('--seq',
                        type=str,
                        default='jiggle_on_toes',
                        help='sequence name')
    parser.add_argument('--sid',
                        type=str,
                        default='50004',
                        choices=sids,
                        help='subject id')
    args = parser.parse_args()

    path = join(args.path, '%s_%s.hdf5' % (args.sid, args.seq))
    if exists(path):
        parse_masks(path)
    else:
        print('Unable to find file %s' % path)