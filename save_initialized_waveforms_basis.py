import visionloader as vl

import pickle
import argparse

from lib.util_fns import shift_align_abs_peak, bspline_upsample_waveforms

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Select initial basis waveforms')

    parser.add_argument('data_pickle', type=str, help='path to data pickle')
    parser.add_argument('basis_pickle', type=str, help='path to input pickle file')
    parser.add_argument('--cell_id', '-c', type=int, help='cell id')
    parser.add_argument('--electrodes', '-e', type=int, nargs='+', help='electrodes')
    parser.add_argument('--upsample', '-u', type=int, default=5, help='upsample factor')
    parser.add_argument('--before', '-b', type=int, default=100, help='left shift samples')
    parser.add_argument('--after', '-a', type=int, default=100, help='right shift samples')
    parser.add_argument('--thresh', '-t', type=float, default=5.0, help='EI amplitude cutoff')
    parser.add_argument('--alignment_sample', '-l', type=int, default=150, help='sample to align peak at')

    args = parser.parse_args()

    shifts = (-args.before, args.after)

    print("Loading data")
    with open(args.data_pickle, 'rb') as pfile:
        preprocessed_dict = pickle.load(pfile)

    eis_by_cell_id = preprocessed_dict['eis_by_cell_id']  # type: Dict[int, np.ndarray]
    cell_list = list(eis_by_cell_id.keys())

    ei_for_template = eis_by_cell_id[args.cell_id]
    basis_waveforms = ei_for_template[list(args.electrodes), :]

    # now we have to upsample and shift the basis waveforms
    bspline_supersampled = bspline_upsample_waveforms(basis_waveforms, args.upsample)
    padded_basis = np.pad(bspline_supersampled,
                          [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
                          mode='constant')

    padded_magnitude = np.linalg.norm(padded_basis, axis=1)
    padded_basis_normed = padded_basis / padded_magnitude[:, None]

    aligned_basis = shift_align_abs_peak(padded_basis_normed, args.alignment_sample)

    with open(args.basis_pickle, 'wb') as pfile:
        pickle_dict = {
            'basis': aligned_basis,
            'upsample': args.upsample,
            'before': args.before,
            'after': args.after,
            'thresh': args.thresh
        }
        pickle.dump(pickle_dict, pfile)
