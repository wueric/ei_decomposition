import visionloader as vl
import electrode_map as el_map

import numpy as np

import pickle

import lib.ei_decomposition as ei_decomp
import lib.spatial_prior_amplitude_time_opt as spat_decomp
import lib.util_fns as util_fns

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='debug coordinate descent plumbing')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('cell_type', type=str, help='cell type of interest')
    parser.add_argument('init_fit')
    args = parser.parse_args()

    print("Loading data")
    dataset = vl.load_vision_data(args.ds_path,
                                  args.ds_name,
                                  include_params=True,
                                  include_ei=True)
    dataset_el_map = dataset.get_electrode_map()
    dataset_adjacency_map = el_map.get_litke_array_adj_mat_by_array_id(dataset.array_id)

    with open(args.init_fit, 'rb') as pfile:
        _ = pickle.load(pfile)
        initial_prefit = pickle.load(pfile)

    initial_prefit_decomp = initial_prefit['decomposition']

    # load the EIs, pack and then unpack and check allclose
    # FIXME do later
    cells_of_type = dataset.get_all_cells_of_type(args.cell_type)
    eis_of_type = {cell_id : dataset.get_ei_for_cell(cell_id).ei for cell_id in cells_of_type}

    max_selected_electrodes, selected_electrodes_by_cell = util_fns.grab_above_threshold_electrodes_and_order(
        eis_of_type,
        5.0
    )

    padded_ei_matrix, electrode_idx_mat, last_valid_indices = util_fns.make_electrode_padded_ei_data_matrix(
        eis_of_type,
        cells_of_type,
        max_selected_electrodes,
        selected_electrodes_by_cell
    )

    flat_matrix_waveforms = util_fns.pack_by_cell_into_flat(padded_ei_matrix, last_valid_indices)
    unflat_matrix = util_fns.unpack_flat_into_by_cell(flat_matrix_waveforms, last_valid_indices)

    assert np.allclose(unflat_matrix, padded_ei_matrix)

    # load existing decomposition, pack, unpack, and check if they're all the same or not
    test_cell_order = list(initial_prefit_decomp.keys()) # type: List[int]

    packed_amplitudes, packed_phases = util_fns.pack_full_by_cell_into_matrix_by_cell(initial_prefit_decomp,
                                                                                      test_cell_order,
                                                                                      max_selected_electrodes,
                                                                                      selected_electrodes_by_cell)

    original_decomp_reinflated = util_fns.pack_by_cell_amplitudes_and_phases_into_ei_shape(
        packed_amplitudes,
        packed_phases,
        selected_electrodes_by_cell,
        cells_of_type,
        512
    )

    print(cells_of_type)
    for cell_id in cells_of_type:
        assert cell_id in initial_prefit_decomp
        assert cell_id in original_decomp_reinflated

        orig_decomp_amplitudes, orig_decomp_phases = initial_prefit_decomp[cell_id]
        reinflated_amplitudes, reinflated_phases = original_decomp_reinflated[cell_id]

        assert np.allclose(orig_decomp_amplitudes, reinflated_amplitudes)
        assert np.allclose(orig_decomp_phases, reinflated_phases)
    print('done')





