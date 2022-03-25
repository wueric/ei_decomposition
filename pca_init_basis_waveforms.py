import numpy as np
from lib.util_fns import pack_significant_electrodes_into_matrix, shift_align_abs_peak, bspline_upsample_waveforms

import pickle
import argparse

from sklearn.mixture import GaussianMixture

from typing import Dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Select initial basis waveforms')

    parser.add_argument('data_pickle', type=str, help='path to data pickle')
    parser.add_argument('basis_pickle', type=str, help='path to output pickle file')
    parser.add_argument('--thresh', '-t', type=float, default=5.0, help='EI amplitude cutoff')
    parser.add_argument('--alignment_sample', '-l', type=int, default=60,
                        help='sample to align peak at, in units of upsampled/padded samples')
    parser.add_argument('--nbasis', '-n', type=int, default=3, help='number of basis waveforms')
    parser.add_argument('--n_pca_components', '-p', type=int, default=5, help='number of PCA components to use for clustering')

    args = parser.parse_args()

    n_pca_components = args.n_pca_components
    n_basis_waveforms = args.nbasis

    with open(args.data_pickle, 'rb') as pfile:

        preprocessed_dict = pickle.load(pfile)

    eis_by_cell_id = preprocessed_dict['eis_by_cell_id'] # type: Dict[int, np.ndarray]
    cell_list = list(eis_by_cell_id.keys())

    padded_channels_sufficient_magnitude, matrix_indices_by_cell_id = pack_significant_electrodes_into_matrix(eis_by_cell_id,
                                                                                     cell_list,
                                                                                     args.thresh)

    if padded_channels_sufficient_magnitude.shape[0] > 5000:
        padded_channels_sufficient_magnitude = padded_channels_sufficient_magnitude[::5,:]

    padded_magnitude = np.linalg.norm(padded_channels_sufficient_magnitude, axis=1)
    padded_channels_normed = padded_channels_sufficient_magnitude / padded_magnitude[:, None]

    aligned_data = shift_align_abs_peak(padded_channels_normed, args.alignment_sample)
    n_waveforms, n_timepoints = aligned_data.shape

    u, s, vh = np.linalg.svd(aligned_data, full_matrices=False)
    v = vh.T

    v_section = v[:, :n_pca_components]
    projection = aligned_data @ v_section

    gm = GaussianMixture(n_components=n_basis_waveforms).fit(projection)
    cluster_values = gm.predict(projection)

    cluster_means = np.zeros((n_basis_waveforms, n_timepoints), dtype=np.float32)
    aligned_data_unscaled = aligned_data * padded_magnitude[:, None]

    for idx in range(n_basis_waveforms):
        selector = (cluster_values == idx)
        cluster_means[idx,:] = np.mean(aligned_data_unscaled[selector, :], axis=0)

    cluster_means = cluster_means / np.linalg.norm(cluster_means, axis=1, keepdims=True)

    pickle_dict = {
        'basis': cluster_means,
        'alignment_sample': args.alignment_sample,
        'before': preprocessed_dict['before'],
        'after': preprocessed_dict['after'],
        'upsample': preprocessed_dict['upsample']
    }

    with open(args.basis_pickle, 'wb') as pfile:
        pickle.dump(pickle_dict, pfile)

