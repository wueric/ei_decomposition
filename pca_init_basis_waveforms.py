import numpy as np
from lib.util_fns import pack_significant_electrodes_into_matrix, generate_fourier_phase_shift_matrices, \
    shift_align_abs_peak, bspline_upsample_waveforms

import pickle
import argparse

from sklearn.mixture import GaussianMixture

import visionloader as vl

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Select initial basis waveforms')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('basis_pickle', type=str, help='path to output pickle file')
    parser.add_argument('cell_type', type=str, help='type of cell to fetch')
    parser.add_argument('--upsample', '-u', type=int, default=5, help='upsample factor')
    parser.add_argument('--before', '-b', type=int, default=100, help='left shift samples')
    parser.add_argument('--after', '-a', type=int, default=100, help='right shift samples')
    parser.add_argument('--thresh', '-t', type=float, default=5.0, help='EI amplitude cutoff')
    parser.add_argument('--alignment_sample', '-l', type=int, default=150, help='sample to align peak at')
    parser.add_argument('--nbasis', '-n', type=int, default=3, help='number of basis waveforms')
    parser.add_argument('--n_pca_components', '-p', type=int, default=5, help='number of PCA components to use for clustering')

    args = parser.parse_args()

    shifts = (-args.before, args.after)
    n_pca_components = args.n_pca_components
    n_basis_waveforms = args.nbasis

    dataset = vl.load_vision_data(args.ds_path,
                                  args.ds_name,
                                  include_params=True,
                                  include_ei=True)

    cell_list = dataset.get_all_cells_of_type(args.cell_type)
    eis_by_cell_id = {cell_id: dataset.get_ei_for_cell(cell_id).ei for cell_id in cell_list}

    ei_data_mat, matrix_indices_by_cell_id = pack_significant_electrodes_into_matrix(eis_by_cell_id,
                                                                                     cell_list,
                                                                                     args.thresh)

    bspline_supersampled = bspline_upsample_waveforms(ei_data_mat, args.upsample)
    padded_channels_sufficient_magnitude = np.pad(bspline_supersampled,
                                                  [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
                                                  mode='constant')

    padded_magnitude = np.linalg.norm(padded_channels_sufficient_magnitude, axis=1)
    padded_channels_normed = padded_channels_sufficient_magnitude / padded_magnitude[:, None]

    aligned_data = shift_align_abs_peak(padded_channels_normed, args.alignment_sample)

    n_waveforms, n_timepoints = aligned_data.shape

    u, s, vh = np.linalg.svd(aligned_data)
    v = vh.T

    v_section = v[:, :n_pca_components]
    projection = aligned_data @ v_section

    gm = GaussianMixture(n_components=n_basis_waveforms).fit(projection)
    cluster_values = gm.predict(projection)

    cluster_means = np.zeros((n_basis_waveforms, n_timepoints), dtype=np.float32)
    aligned_data_unscaled = aligned_data * padded_magnitude[:, None]

    for idx in range(n_basis_waveforms):
        selector = (cluster_values == idx)
        cluster_means[idx,:] = np.mean(aligned_data_unscaled[selector, :])

    cluster_means = cluster_means / np.linalg.norm(cluster_means, axis=1, keepdims=True)

    with open(args.basis_pickle, 'wb') as pfile:
        pickle_dict = {
            'basis' : cluster_means
        }
        pickle.dump(pickle_dict, pfile)





