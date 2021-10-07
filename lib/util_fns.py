from collections import namedtuple

import numpy as np
from scipy import interpolate as interpolate

from typing import List, Dict, Union, Tuple, Sequence, Optional

EIDecomposition = namedtuple('EIDecomposition', ['amplitude', 'delay'])
UnsharedBasisEIDecomposition = namedtuple('UnsharedBasisEIDecomposition', ['amplitude', 'delay', 'basis'])


def bspline_upsample_waveforms(waveforms: np.ndarray,
                               upsample_factor: int) -> np.ndarray:
    '''

    :param waveform: shape (n_waveforms, n_samples) original waveforms
    :return: upsampled waveforms, shape (n_waveforms, n_shifts, n_samples)
    '''

    n_waveforms, n_orig_samples = waveforms.shape
    upsampled = np.zeros((n_waveforms, n_orig_samples * upsample_factor))

    orig_time_samples = np.r_[0:n_orig_samples]  # shape (n_samples, )
    upsample_timepoints = np.linspace(0, n_orig_samples, n_orig_samples * upsample_factor)

    for idx in range(n_waveforms):
        orig_waveform_1d = waveforms[idx, :]
        bspline = interpolate.splrep(orig_time_samples, orig_waveform_1d)

        waveform_shifted = interpolate.splev(upsample_timepoints, bspline)
        # shape (n_shifts, n_orig_samples)
        upsampled[idx, :] = waveform_shifted

    return upsampled


def bspline_upsample_waveforms_padded_by_cell(waveforms_by_cell: np.ndarray,
                                              last_valid_indices: np.ndarray,
                                              upsample_factor: int) -> np.ndarray:
    '''
    Upsample waveforms with bspline interpolation, where the waveforms are grouped by cell
        and some of the electrodes are disused
    :param waveforms_by_cell: shape (n_cells, n_max_electrodes, n_timepoints)
    :param last_valid_indices: integer, shape (n_cells, ), index of the last valid electrode for each cell
        in waveforms_by_cell
    :param upsample_factor: upsample factor
    :return: upsampled waveforms, shape (n_cells, n_max_electrodes, n_timepoints * upsample_factor)
    '''

    n_cells, max_n_electrodes, n_orig_samples = waveforms_by_cell.shape
    upsampled = np.zeros((n_cells, max_n_electrodes, n_orig_samples * upsample_factor), dtype=np.float32)

    orig_time_samples = np.r_[0:n_orig_samples]
    upsample_timepoints = np.linspace(0, n_orig_samples, n_orig_samples * upsample_factor)

    for cell_idx in range(n_cells):
        max_valid_idx = last_valid_indices[cell_idx]
        for el_idx in range(max_valid_idx):
            orig_waveform = waveforms_by_cell[cell_idx, el_idx, :]
            bspline = interpolate.splrep(orig_time_samples, orig_waveform)

            waveform_shifted = interpolate.splev(upsample_timepoints, bspline)
            # shape (n_shifts, n_orig_samples)
            upsampled[cell_idx, el_idx, :] = waveform_shifted

    return upsampled


def grab_above_threshold_electrodes_and_order(eis_by_cell_id: Dict[int, np.ndarray],
                                              threshold: float) -> Tuple[int, Dict[int, np.ndarray]]:
    '''
    Grabs indices of electrodes that are above threhsolds

    :param eis_by_cell_id: key cell_id (int) -> value EI matrix (np.ndarray)
    :param threshold: cutoff for including electrode in EI, maximum amplitude of EI must exceed this threshold
    :return: maximum number of electrodes included; Dict where key is cell_id, value is np.ndarray with shape
        (n_electrodes_included, ) containing the indices of the above threshold electrodes. Order in this array
        is important
    '''
    above_threshold_index_dict = {}  # type: Dict[int, np.ndarray]
    max_threshold_electrodes = -np.inf

    for cell_id, full_ei in eis_by_cell_id.items():
        max_amplitude = np.amax(np.abs(full_ei), axis=1)
        exceeds_threshold = max_amplitude > threshold
        n_exceeds_threshold = np.sum(exceeds_threshold)
        max_threshold_electrodes = max(max_threshold_electrodes, n_exceeds_threshold)

        above_threshold_index_dict[cell_id] = np.squeeze(np.argwhere(exceeds_threshold))

    return max_threshold_electrodes, above_threshold_index_dict


def make_electrode_padded_ei_data_matrix(eis_by_cell_id: Dict[int, np.ndarray],
                                         cell_order: List[int],
                                         max_n_electrodes: int,
                                         electrode_selection_by_cell: Dict[int, np.ndarray]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Given a previously determined selection of electrodes for each cell,
        pack a matrix representing the reduced EI for all of the cells

    :param eis_by_cell_id: Dict[int - cell_id, np.ndarray - EI with shape (n_electrodes, n_timepoints)
    :param cell_order : List[int] list of cell id
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray], first one has shape (n_cells, n_electrodes_max, n_timepoints)
        corresponding to the padded EIs, and second one is integer valued with shape (n_cells, n_electrodes_max),
        where unused slots are given value -1. Third has shape (n_cells, ) and is integer valued.
        It contains the index of the last valid electrode in the padding
    '''

    n_cells = len(eis_by_cell_id)
    n_timepoints = eis_by_cell_id[cell_order[0]].shape[1]

    padded_eis = np.zeros((n_cells, max_n_electrodes, n_timepoints), dtype=np.float32)
    electrode_idx_mat = -np.ones((n_cells, max_n_electrodes), dtype=np.int32)
    last_valid_idx = np.zeros((n_cells,), dtype=np.int32)

    for i, cell_id in enumerate(cell_order):
        full_ei = eis_by_cell_id[cell_id]

        channels_included = electrode_selection_by_cell[cell_id]
        n_channels_included = channels_included.shape[0]

        padded_eis[i, :n_channels_included, :] = full_ei[channels_included, :]
        electrode_idx_mat[i, :n_channels_included] = channels_included

        last_valid_idx[i] = n_channels_included

    return padded_eis, electrode_idx_mat, last_valid_idx


def pack_full_by_cell_into_matrix_by_cell(decomposition_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
                                          cell_order: List[int],
                                          max_n_electrodes: int,
                                          electrode_selection_by_cell_id: Dict[int, np.ndarray]) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''

    Packs an existing decomposition (in full EI form factor) into the stacked by-cell tensor representation
        that the spatial continuity decomposition algorithm uses

    :param decomposition_dict: existing decomposition dict, key is integer cell_id, values are Tuple of
        np.ndarray, first entry in tuple is previously fit decomposition weight, shape (n_electrodes, n_basis_vectors),
        second entry in tuple is previously fit decomposition delay, shape (n_electrodes, n_basis_vectors)
    :param cell_order: ordering of cells that we want in the output matrix, list of cell_id
    :param max_n_electrodes: maximum number of above threshold electrodes that we are including
    :param electrode_selection_by_cell_id: selection of nonzero electrodes, key is cell_id, value is np.ndarray used
        to slice the provided decomposition to grab fits for electrodes that we care about. Order in the np.ndarray
        matters
    :return: packed decomposition tensor, shape (n_cells, max_n_electrodes, n_basis_waveforms), and packed phase tensor,
        shape (n_cells, max_n_electrodes, n_basis_waveforms)
    '''

    _, n_basis_waveforms = decomposition_dict[cell_order[0]][0].shape
    n_cells = len(cell_order)

    stacked_amplitudes = np.zeros((n_cells, max_n_electrodes, n_basis_waveforms), dtype=np.float32)
    stacked_phases = np.zeros((n_cells, max_n_electrodes, n_basis_waveforms), dtype=np.int32)

    for idx, cell_id in enumerate(cell_order):
        init_decomp_amp, init_decomp_phase = decomposition_dict[cell_id]
        selected_electrodes_in_order = electrode_selection_by_cell_id[cell_id]
        n_electrodes_selected = selected_electrodes_in_order.shape[0]

        stacked_amplitudes[idx, :n_electrodes_selected, :] = init_decomp_amp[selected_electrodes_in_order, :]
        stacked_phases[idx, :n_electrodes_selected, :] = init_decomp_phase[selected_electrodes_in_order, :]

    return stacked_amplitudes, stacked_phases


def one_pad_disused_by_cell(magnitudes_arranged_by_cell: np.ndarray,
                            last_valid_indices: np.ndarray) -> np.ndarray:
    '''
    Replaces entries corresponding to unused electrodes with 1
        Useful to avoid dividing by zero

    :param magnitudes_arranged_by_cell: magnitudes of each channel for each cell
        shape (n_cells, max_n_electrodes)
    :param last_valid_indices: shape (n_cells, ) integer valued, contains the number
        of valid electrodes by cell in magnitudes_arranged_by_cell
    :return: (n_cells, max_n_electrodes), where entries corresponding to unused electrodes
        are replaced by 1 for safe multiplication and division
    '''

    n_cells, max_n_electrodes = magnitudes_arranged_by_cell.shape
    output_magnitudes = np.ones_like(magnitudes_arranged_by_cell, dtype=np.float32)
    for cell_idx in range(n_cells):
        top_idx = last_valid_indices[cell_idx]
        output_magnitudes[cell_idx, :top_idx] = magnitudes_arranged_by_cell[cell_idx, :top_idx]

    return output_magnitudes


def pack_by_cell_into_flat(waveforms_arranged_by_cell: np.ndarray,
                           last_valid_indices: np.ndarray) -> np.ndarray:
    '''
    Rearranges and flattens 3D matrix of time domain waveforms that are arranged by cell into
        a 2D matrix of time domain waveforms. Automatically cuts out zero-valued waveforms

    :param waveforms_arranged_by_cell: shape (n_cells, n_max_electrodes, ...)
    :param last_valid_indices: shape (n_cells, ), integer valued, contains the index of the last
        valid electrode in waveforms_arranged_by_cell
    :return: shape (n_waveforms_total, ...)
    '''

    n_cells, n_max_electrodes = waveforms_arranged_by_cell.shape[:2]
    n_legit_waveforms = np.sum(last_valid_indices)

    remaining_dimensions = waveforms_arranged_by_cell.shape[2:]
    output_flat_matrix_dims = [n_legit_waveforms, ]
    output_flat_matrix_dims.extend(list(remaining_dimensions))

    output_flat_matrix = np.zeros(output_flat_matrix_dims, dtype=waveforms_arranged_by_cell.dtype)

    write_offset = 0
    for cell_idx in range(n_cells):
        n_electrodes_for_cell = last_valid_indices[cell_idx]
        write_end = write_offset + n_electrodes_for_cell
        output_flat_matrix[write_offset:write_end, ...] = waveforms_arranged_by_cell[cell_idx, :n_electrodes_for_cell,
                                                          ...]

        write_offset = write_end

    return output_flat_matrix


def unpack_flat_into_by_cell(flat_matrix: np.ndarray,
                             last_valid_indices: np.ndarray) -> np.ndarray:
    '''
    Rearranges and unflattens a flat 2D matrix of time domain waveforms into a padded 3D matrix of time domain waveforms
        tht are arranged by cell

    Reverse of pack_by_cell_into_flat

    :param flat_matrix: shape (n_waveforms_total, n_timepoints), flat matrix of waveforms
    :param last_valid_indices:  shape (n_cells, ), integer valued, contains the index of the last valid electrode for
        each cell
    :return: shape (n_cells, n_max_electrodes, n_timepoints)
    '''

    n_total_waveforms, n_timepoints = flat_matrix.shape
    n_cells = last_valid_indices.shape[0]
    n_max_electrodes = np.max(last_valid_indices)

    waveforms_padded_by_cell = np.zeros((n_cells, n_max_electrodes, n_timepoints), dtype=np.float32)

    read_offset = 0
    for cell_idx in range(n_cells):
        n_waveforms_to_get = last_valid_indices[cell_idx]
        read_end = n_waveforms_to_get + read_offset

        waveforms_padded_by_cell[cell_idx, :n_waveforms_to_get, :] = flat_matrix[read_offset:read_end, :]
        read_offset = read_end

    return waveforms_padded_by_cell


def get_neighborhood_indices_from_adj_mat_dfs(by_cell_adj_mat: np.ndarray,
                                              center_electrode_idx: int,
                                              search_depth: int) -> np.ndarray:
    '''
    Get the indices of the nearest neighbors of a center electrode, as well as their nearest
        neighbors from the included-electrodes-only adjacency matrix representation

    Indices in the adjacency list correspond to the indices of the included-electrode-only adjacency matrix
    :param by_cell_adj_mat:
    :param center_electrode_idx:
    :param search_depth: depth in the DFS for which we search the neighborhood
    :return: shape (n_cells, ?) ragged np.ndarray of integer
    '''

    def iterative_dfs_helper(adj_mat: np.ndarray,
                             root_vert: int,
                             depth: int) -> List[int]:

        dfs_stack = [(root_vert, depth), ]
        visited = set()
        neighborhood_list = []

        while len(dfs_stack) != 0:

            # visit the node
            current_node, current_depth = dfs_stack.pop()
            visited.add(current_node)
            neighborhood_list.append(current_node)

            if current_depth != 0:
                neighbor_els, = np.nonzero(adj_mat[current_node, :])
                for neighbor_el in neighbor_els:
                    if neighbor_el not in visited:
                        dfs_stack.append((neighbor_el, current_depth - 1))

        return neighborhood_list

    n_cells = by_cell_adj_mat.shape[0]
    output_adj_lists = np.empty((n_cells,), dtype=np.object)

    for cell_idx in range(n_cells):
        neighborhood_list = iterative_dfs_helper(by_cell_adj_mat[cell_idx, :, :], center_electrode_idx, search_depth)
        output_adj_lists[cell_idx] = neighborhood_list

    return output_adj_lists


def make_spatial_neighbors_mean_matrix(raw_adjacency_mat: np.ndarray,
                                       included_in_padded_ei: np.ndarray,
                                       last_valid_indices: np.ndarray) -> np.ndarray:
    '''
    Converts adjacency list into a series of adjacency mean matrices, each corresponding
        to a particular cell

    Rules for dealing with edges and excluded electrodes:
        (1) If an electrode is at the edge of a cell (i.e. at least one of its neighboring electrodes is excluded
            in the optimization calculation because its amplitude is insufficient), we still include those excluded
            neighbors in the denominator for the mean calculation because zero is still a meaningful signal
        (2) Edges of the array will not be included in the mean calculation (i.e. if an electrode has fewer neighbors
            than typical because it is located at the edge of the electrode array, the number of neighbors is
            the number of existing real neighbors rather than the number of typical neighbors)

    :param raw_adjacency_mat: adjacency list, shape (n_electrodes, ?), ragged np.ndarray
    :param included_in_padded_ei: shape (n_cells, n_max_electrodes), electrode order for each padded ei
        where disused electrodes are marked with -1
    :param last_valid_indices: shape (n_cells, ), integer, corresponding to the last valid col
        of included_in_padd_ei for each cell
    :return: np.ndarray, shape (n_cells, n_max_electrodes, n_max_electrodes), corresponding to the neighbor
        mean matrix
    '''

    n_cells, n_max_electrodes = included_in_padded_ei.shape

    # first build the adjacency matrix for the full electrode map
    # from that then calculate the mean matrix for the full electrode map
    # then we take submatrices from the full mean matrix for each cell
    full_matrix_n_electrodes = raw_adjacency_mat.shape[0]
    full_adj_mat = np.zeros((full_matrix_n_electrodes, full_matrix_n_electrodes), dtype=np.float32)
    for center_idx in range(full_matrix_n_electrodes):
        nn_indices = raw_adjacency_mat[center_idx]
        full_adj_mat[center_idx, nn_indices] = 1.0

    # the mean matrix is calculated by dividing by columnwise sums
    adj_mat_csums = np.sum(full_adj_mat, axis=0)  # shape (full_matrix_n_electrodes, full_matrix_n_electrodes)
    full_mean_mat = full_adj_mat / adj_mat_csums[None, :]  # shape (full_matrix_n_electrodes, full_matrix_n_electrodes)

    cell_mean_matrix = np.zeros((n_cells, n_max_electrodes, n_max_electrodes), dtype=np.float32)
    for cell_idx in range(n_cells):
        n_valid_electrodes_cell = last_valid_indices[cell_idx]

        relevant_electrodes_indices = included_in_padded_ei[cell_idx][:n_valid_electrodes_cell]
        relevant_submatrix = full_mean_mat[np.ix_(relevant_electrodes_indices, relevant_electrodes_indices)]
        cell_mean_matrix[cell_idx, :n_valid_electrodes_cell, :n_valid_electrodes_cell] = relevant_submatrix

    return cell_mean_matrix


def generate_fourier_phase_shift_matrices(sample_delays: np.ndarray,
                                          n_frequencies: int) -> np.ndarray:
    '''
    Generate complex-valued sample delay matrices corresponding
        to integer sample delays

    When we take the Fourier transform of the canonical waveforms, which
        have shape (n_canonical_waveforms, n_samples), we get a transform
        vector with shape (n_canonical_waveforms, n_frequencies = n_samples)

    We want to multiply element-wise each entry in the Fourier transform by
        the corresponding phase shift e^{-j 2 * pi * tau * f / F} and broadcast
        accordingly to account of the different number of shifts

    :param sample_delays: integer, shape (n_observations, n_canonical_waveforms)
    :param n_frequencies: number of frequencies, equal to number of samples
    :return: phase delay matrix, complex valued,
        shape (n_observations, n_canonical_waveforms, n_frequencies)
    '''
    sample_delays_slice = [slice(None) for _ in range(sample_delays.ndim)]
    sample_delays_slice.append(None)
    sample_delays_slice = tuple(sample_delays_slice)

    phase_radians_slice = [None for _ in range(sample_delays.ndim)]
    phase_radians_slice.append(slice(None))
    phase_radians_slice = tuple(phase_radians_slice)

    # this has value f/F
    frequencies = np.fft.rfftfreq(n_frequencies)  # shape (n_frequencies, )

    # this is 2 * pi * f / F
    phase_radians = 2 * np.pi * frequencies  # shape (n_frequencies, )

    # this is -j * 2 * pi * tau * f / F
    complex_exponential_argument = -1j * sample_delays[sample_delays_slice] * phase_radians[phase_radians_slice]

    return np.exp(complex_exponential_argument)


def auto_prebatch_pack_significant_electrodes(eis_by_cell_id: Dict[int, np.ndarray],
                                              snr_abs_threshold: Union[float, int],
                                              batch_tot_els : int = 5000) \
        -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    '''

    We use this function when we want each cell to have its own set of basis waveforms

    :param eis_by_cell_id: cell_id -> np.ndarray of shape (n_electrodes, n_timepoints),
        corresponding to the EI for every cell that we want to pool together
    :param snr_abs_threshold: threshold, waveforms with maximum amplitude below this value
        are discarded
    :return:
        np.ndarray of shape (batch, max_n_sig_electrodes, n_timepoints), containing the data matrices

        boolean np.ndarray of shape (batch, max_n_sig_electrodes), where each entry that corresponds to real
        data is marked True, and each entry that corresponds to no-data (i.e. padding) is marked False

        boolean np.ndarray of shape (batch, n_electrodes_total), where each entry corresponds to the integer
        index for which channel the given waveform came from. Needed to reconstruct the decomposition
    '''

    n_els_total, n_timepoints = eis_by_cell_id[list(eis_by_cell_id.keys())[0]].shape

    initial_order, acc_by_ei = [], []
    for cell_id, ei_mat in eis_by_cell_id.items():
        initial_order.append(cell_id)
        acc_by_ei.append(ei_mat)

    initial_order = np.array(initial_order, dtype=np.int32) # shape (n_cells, )
    ei_stacked_mat = np.stack(acc_by_ei, axis=0) # shape (n_cells, n_electrodes, n_timepoints)
    exceeds_thresh = np.max(ei_stacked_mat, axis=2) > snr_abs_threshold # shape (n_cells, n_electrodes)
    n_exceeds_thresholds_cell = np.sum(exceeds_thresh, axis=1) # shape (n_cells, ), integer valued

    sorted_by_elcount = np.argsort(n_exceeds_thresholds_cell) # shape (n_cells, )

    sorted_elcount = n_exceeds_thresholds_cell[sorted_by_elcount] # shape (n_cells, )
    sorted_cell_id_order = initial_order[sorted_by_elcount] # shape (n_cells, )
    exceeds_thresh_sorted_order = exceeds_thresh[sorted_by_elcount, :] # shape (n_cells, n_electrodes)
    ei_stacked_sorted_order = ei_stacked_mat[sorted_by_elcount,:, :] # shpae (n_cells, n_electrodes, n_timepoints)

    n_cells = sorted_cell_id_order.shape[0]

    autobatched_list = [] # type: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    batch_start, i = 0, 1
    while batch_start < n_cells:

        while i <= n_cells and (sorted_elcount[i-1] * (batch_start - i)) < batch_tot_els:
            i += 1

        # now we group all electrodes between batch_start and i+1
        batch_size = i - batch_start
        batch_n_els = sorted_elcount[i-2]

        batch_waveforms = np.zeros((batch_size, batch_n_els, n_timepoints), dtype=np.float32)
        batch_valid_els = np.zeros((batch_size, batch_n_els), dtype=bool)
        batch_recovery_idx = np.zeros((batch_size, n_els_total), dtype=np.int32)

        for write_idx, read_idx in enumerate(range(batch_start, i)):

            n_valid_cell = sorted_elcount[read_idx]
            exceeds_thresh_cell_selector = exceeds_thresh_sorted_order[read_idx,:] # shape (n_els_total, ) bool-valued

            batch_waveforms[write_idx, :n_valid_cell, :] = ei_stacked_sorted_order[read_idx, exceeds_thresh_cell_selector, :]
            batch_valid_els[write_idx, :n_valid_cell] = True
            batch_recovery_idx[write_idx, :] = exceeds_thresh_cell_selector

        batch_cell_ids = sorted_cell_id_order[batch_start:i]

        autobatched_list.append((batch_waveforms, batch_valid_els, batch_recovery_idx, batch_cell_ids))

        batch_start = i

    return autobatched_list


def auto_unbatch_unpack_significant_electrodes(batched_amplitude_phase_list : List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                               batched_raw_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) \
        -> Dict[int, UnsharedBasisEIDecomposition]:
    '''

    :param batched_amplitude_phase_list:
    :param batched_raw_data:
    :return:
    '''

    if len(batched_amplitude_phase_list) != len(batched_raw_data):
        raise ValueError('Batched decomposition list must have same length as batched raw data')

    decomp_dict = {} # type: Dict[int, UnsharedBasisEIDecomposition]
    for decomp_tuple, data_tuple in zip(batched_amplitude_phase_list, batched_raw_data):

        # batched_amplitudes: shape (batch, n_max_els, n_basis), real floating point valued
        # batched_phases: shape (batch, n_max_els, n_basis), integer-valued
        # batched_basis: shape (batch, n_basis, n_timepoints), real floating point valued
        batched_amplitudes, batched_phases, batched_basis = decomp_tuple

        # batch_valid_els: shape (batch, n_max_els), boolean-valued
        # batch_recovery_indices: shape (batch, n_tot_els), integer-valued
        # batch_cell_ids: shape (batch, ), integer-valued cell IDs in order
        _, batch_valid_els, batch_recovery_indices, batch_cell_ids = data_tuple

        batch_size, n_tot_els = batch_recovery_indices.shape
        _, n_basis, n_timepoints = batched_basis.shape
        for idx in range(batch_size):
            reinflated_amplitudes = np.zeros((n_tot_els, n_basis), dtype=np.float32)
            reinflated_shifts = np.zeros((n_tot_els, n_basis), dtype=np.int32)

            write_indices = batch_recovery_indices[idx, :]
            read_indices = batch_valid_els[idx, :]

            reinflated_amplitudes[write_indices, :] = batched_amplitudes[idx, read_indices, :]
            reinflated_shifts[write_indices, :] = batched_phases[idx, read_indices, :]
            reinflated_basis = batched_basis[idx, :, :]

            cell_id = batch_cell_ids[idx]

            decomp_dict[cell_id] = UnsharedBasisEIDecomposition(reinflated_amplitudes, reinflated_shifts,
                                                                reinflated_basis)

    return decomp_dict


def batched_pack_significant_electrodes(eis_by_cell_id: Dict[int, np.ndarray],
                                        cell_order: List[int],
                                        snr_abs_threshold: Union[float, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''
    Packs the waveforms from each cell into a batched data matrix, where the batch
        dimension corresponds to the number of cells

    IMPORTANT: TREATS EACH CELL SEPARATELY.

    We use this function when we want each cell to have its own set of basis waveforms

    :param eis_by_cell_id: cell_id -> np.ndarray of shape (n_electrodes, n_timepoints),
        corresponding to the EI for every cell that we want to pool together
    :param cell_order: List of cell_id, corresponding to the order of the cells. We do
        all of our analyses in this order
    :param snr_abs_threshold: threshold, waveforms with maximum amplitude below this value
        are discarded
    :return:
        np.ndarray of shape (batch, max_n_sig_electrodes, n_timepoints), containing the data matrices

        boolean np.ndarray of shape (batch, max_n_sig_electrodes), where each entry that corresponds to real
        data is marked True, and each entry that corresponds to no-data (i.e. padding) is marked False

        boolean np.ndarray of shape (batch, n_electrodes_total), where each entry corresponds to the integer
        index for which channel the given waveform came from. Needed to reconstruct the decomposition
    '''

    n_cells = len(cell_order)
    n_electrodes, n_timepoints = eis_by_cell_id[list(eis_by_cell_id.keys())[0]].shape

    to_stack = [] # type: List[np.ndarray]
    n_valid = [] # type: List[int]

    indices_selected = np.zeros((n_cells, n_electrodes), dtype=bool)
    max_n_sig_els = 0

    for i, cell_id in enumerate(cell_order):
        ei_mat = eis_by_cell_id[cell_id]

        chans_sufficient_magnitude = np.max(np.abs(ei_mat), axis=1) > snr_abs_threshold

        n_chans_sufficient = np.sum(chans_sufficient_magnitude)  # type: int

        n_valid.append(n_chans_sufficient)
        max_n_sig_els = max(n_chans_sufficient, max_n_sig_els)

        to_stack.append(ei_mat[chans_sufficient_magnitude, :])
        indices_selected[i,:] = chans_sufficient_magnitude

    is_valid_matrix = np.zeros((n_cells, max_n_sig_els), dtype=bool)
    batched_data_matrix = np.zeros((n_cells, max_n_sig_els, n_timepoints), dtype=np.float32)

    for i, (selected_data, n_valid_chans) in enumerate(zip(to_stack, n_valid)):
        batched_data_matrix[i, :n_valid_chans, :] = selected_data
        is_valid_matrix[i, :n_valid_chans] = True

    return batched_data_matrix, is_valid_matrix, indices_selected


def batched_unpack_significant_electrodes(batched_amplitudes_matrix : np.ndarray,
                                          batched_phase_matrix: np.ndarray,
                                          batched_basis_waveforms: np.ndarray,
                                          is_valid_matrix: np.ndarray,
                                          index_sel_matrix: np.ndarray,
                                          cell_order : List[int]) -> Dict[int, UnsharedBasisEIDecomposition]:
    '''

    :param batched_amplitudes_matrix: fitted amplitudes, real-valued shape (n_cells, n_observations, n_basis_waveforms)
    :param batched_phase_matrix: fitted phases, integer-valued shape (n_cells, n_observations, n_basis_waveforms)
    :param batched_basis_waveforms: fitted basis waveforms, real-valued shape (n_cells, n_basis_waveforms, n_timepoints)
    :param is_valid_matrix: shape (n_cells, n_observations), 0/1 integer-valued matrix, 1 if valid, 0 if padding
    :param index_sel_matrix: shape (n_cells, n_electrodes_orig), integer-valued index matrix to use to reconstruct
        the original EI matrix
    :param cell_order: order of the cell ids
    :return:
    '''

    n_cells, n_observations, n_basis_waveforms = batched_amplitudes_matrix.shape
    n_orig_els = index_sel_matrix.shape[1]

    output_dict = {} # type: Dict[int, UnsharedBasisEIDecomposition]

    for i, cell_id in enumerate(cell_order):

        put_back_matrix = index_sel_matrix[i,:]
        valid_selector = is_valid_matrix[i,:].astype(bool)

        amplitudes_reinflated = np.zeros((n_orig_els, n_basis_waveforms), dtype=np.float32)
        phases_reinflated = np.zeros((n_orig_els, n_basis_waveforms), dtype=np.int32)

        amplitudes_reinflated[put_back_matrix, :] = batched_amplitudes_matrix[i, valid_selector, :]
        phases_reinflated[put_back_matrix, :] = batched_phase_matrix[i, valid_selector, :]

        output_dict[cell_id] = UnsharedBasisEIDecomposition(amplitudes_reinflated, phases_reinflated,
                                                            batched_basis_waveforms[i, :, :])

    return output_dict


def pack_significant_electrodes_into_matrix(eis_by_cell_id: Dict[int, np.ndarray],
                                            cell_order: List[int],
                                            snr_abs_threshold: Union[float, int]) \
        -> Tuple[np.ndarray, Dict[int, Tuple[slice, Sequence[int]]]]:
    '''
    Packs all the waveforms from all cells that exceed a certain SNR threshold
        into a waveform data matrix.

    IMPORTANT: POOLS ALL OF THE WAVEFORMS FROM THE DIFFERENT CELLS TOGETHER.

    We use this function when we want all of the cells provided as input to share
        a single basis set.

    :param eis_by_cell_id: cell_id -> np.ndarray of shape (n_electrodes, n_timepoints),
        corresponding to the EI for every cell that we want to pool together
    :param cell_order: List of cell_id, corresponding to the order of the cells. We do
        all of our analyses in this order
    :param snr_abs_threshold: threshold, waveforms with maximum amplitude below this value
        are discarded
    :return:

    The combined data matrix, and all of the information needed to undo the pooling operation

    np.ndarray of shape (n_total_waveforms, n_timepoints)

    Dict cell_id to Tuple, each tuple contains
        1. A slice object, enabling us to select the waveforms corresponding to this particular
            cell out of the data matrix
        2. A np.ndarray of boolean, corresponding to the indices of the electrodes that each of the
            waveforms corresponded to

    '''
    matrix_indices_by_cell_id = {}  # type: Dict[int, Tuple[slice, Sequence[int]]]
    to_concat = []  # type: List[np.ndarray]

    cat_low = 0  # type: int
    for cell_id in cell_order:
        ei_mat = eis_by_cell_id[cell_id]

        chans_sufficient_magnitude = np.max(np.abs(ei_mat), axis=1) > snr_abs_threshold
        n_chans_sufficient = np.sum(chans_sufficient_magnitude)  # type: int

        readback_slice = slice(cat_low, cat_low + n_chans_sufficient)
        matrix_indices_by_cell_id[cell_id] = (readback_slice, chans_sufficient_magnitude)
        to_concat.append(ei_mat[chans_sufficient_magnitude, :])

        cat_low += n_chans_sufficient

    ei_data_mat = np.concatenate(to_concat, axis=0)

    return ei_data_mat, matrix_indices_by_cell_id


def unpack_amplitudes_and_phases_into_ei_shape(packed_amplitude_matrix: np.ndarray,
                                               packed_phase_matrix: np.ndarray,
                                               orig_ei_by_cell_id: Dict[int, np.ndarray],
                                               cell_order: List[int],
                                               unpack_slice_dict: Dict[int, Tuple[slice, Sequence[int]]]) \
        -> Dict[int, EIDecomposition]:
    n_observations, n_basis_vectors = packed_amplitude_matrix.shape

    result_dict = {}  # type: Dict[int, EIDecomposition]
    for cell_id in cell_order:
        orig_ei_mat = orig_ei_by_cell_id[cell_id]
        n_channels = orig_ei_mat.shape[0]

        slice_section, sufficient_snr = unpack_slice_dict[cell_id]

        amplitude_matrix = np.zeros((n_channels, n_basis_vectors), dtype=np.float32)
        amplitude_matrix[sufficient_snr, :] = packed_amplitude_matrix[slice_section, :]

        delay_vector = np.zeros((n_channels, n_basis_vectors), dtype=np.int32)
        delay_vector[sufficient_snr, :] = packed_phase_matrix[slice_section, :]

        result_dict[cell_id] = (amplitude_matrix, delay_vector)

    return result_dict


def pack_by_cell_amplitudes_and_phases_into_ei_shape(by_cell_amplitude_matrix: np.ndarray,
                                                     by_cell_phase_matrix: np.ndarray,
                                                     above_threshold_els_by_cell: Dict[int, np.ndarray],
                                                     cell_order: List[int],
                                                     orig_ei_n_electrodes: int) \
        -> Dict[int, EIDecomposition]:
    '''
    Converts the by-cell matrix representation of an EI decomposition into EIDecomposition

    :param by_cell_amplitude_matrix: amplitude matrix, shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    :param by_cell_phase_matrix: phase matrix, integer valued, shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    :param above_threshold_els_by_cell: valid electrodes by cell id, cell id -> np.ndarray of shape (n_electrodes, )
        Each of the np.ndarray may have a different length
    :param cell_order: ordering of cells, list of integers
    :param orig_ei_n_electrodes: number of electrodes included in the full-size EI
    :return: Dict[int, EIDecomposition], EIDecomposition for each cell, keyed by cell_id
    '''

    n_cells, max_n_electrodes, n_basis_waveforms = by_cell_amplitude_matrix.shape

    result_dict = {}  # type: Dict[int, EIDecomposition]
    for idx, cell_id in enumerate(cell_order):
        electrode_order_valid = above_threshold_els_by_cell[cell_id]
        n_valid_electrodes = electrode_order_valid.shape[0]

        amplitude_matrix = np.zeros((orig_ei_n_electrodes, n_basis_waveforms), dtype=np.float32)
        amplitude_matrix[electrode_order_valid, :] = by_cell_amplitude_matrix[idx, :n_valid_electrodes, :]

        delay_matrix = np.zeros((orig_ei_n_electrodes, n_basis_waveforms), dtype=np.int32)
        delay_matrix[electrode_order_valid, :] = by_cell_phase_matrix[idx, :n_valid_electrodes, :]

        result_dict[cell_id] = (amplitude_matrix, delay_matrix)

    return result_dict


def shift_align_abs_peak(normalized_data_matrix: np.ndarray,
                         abs_peak_alignment_point: int) -> np.ndarray:
    # shift all of the waveforms such that their maximum deviation from zero
    # is at the same point
    n_waveforms, n_samples = normalized_data_matrix.shape

    max_point = np.argmax(np.abs(normalized_data_matrix), axis=1)
    delays = abs_peak_alignment_point - max_point

    data_ft = np.fft.rfft(normalized_data_matrix, axis=1)
    shift_matrix = generate_fourier_phase_shift_matrices(delays, n_samples)

    aligned_data = np.real(np.fft.irfft(data_ft * shift_matrix, axis=1, n=n_samples))

    return aligned_data
