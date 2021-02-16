import visionloader as vl
import electrode_map as el_map

import torch

import pickle

import lib.ei_decomposition as ei_decomp
import lib.spatial_prior_amplitude_time_opt as spat_decomp

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='jointly decompose noramlized EIs for a given cell type into constituent waveforms by jointly optimizing waveforms, shifts, and amplitudes')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('cell_type', type=str, help='cell type of interest')
    parser.add_argument('output', type=str, help='path to output pickle file')
    parser.add_argument('--nbasis', '-n', type=int, default=3, help='number of basis waveforms')
    parser.add_argument('--maxiter', '-m', type=int, default=25, help='maximum number of iterations to run')
    parser.add_argument('--weight_reg', '-w', type=float, default=7.5e-2, help='L1 regularization lambda for amplitudes')
    parser.add_argument('--sobolev_reg', '-s', type=float, default=1e-3,
                        help='L2 regularization for waveform second derivatives')
    parser.add_argument('--spatial_reg', '-q', type=float, default=1e-3, help='Spatial continuity regularization lambda')
    parser.add_argument('--upsample', '-u', type=int, default=5, help='upsample factor')
    parser.add_argument('--before', '-b', type=int, default=100, help='left shift samples')
    parser.add_argument('--after', '-a', type=int, default=100, help='right shift samples')
    parser.add_argument('--grid_step', type=int, default=5, help='step size for grid search')
    parser.add_argument('--grid_top_n', type=int, default=4, help='top n for grid search')
    parser.add_argument('--fine_search_width', type=int, default=2, help='width for fine search')
    parser.add_argument('--grid_batch_size', type=int, default=8192, help='grid search batch size')
    parser.add_argument('--thresh', '-t', type=float, default=5.0, help='EI amplitude cutoff')
    parser.add_argument('--cell_list', '-c', type=str, default=None,
                        help='Override cell_type argument, instead use cell ids in specified file')
    parser.add_argument('--renormalize', '-r', action='store_true', default=False, help='renormalize data waveforms')
    parser.add_argument('--select_by_l1', '-d', action='store_true', default=False,
                        help='include L1 regularization when picking best search candidates')
    parser.add_argument('--initialize_basis', '-i', type=str, default=None, help='path to initialized basis')

    args = parser.parse_args()

    compute_device = torch.device('cuda')

    print("Loading data")
    dataset = vl.load_vision_data(args.ds_path,
                                  args.ds_name,
                                  include_params=True,
                                  include_ei=True)
    dataset_el_map = dataset.get_electrode_map()
    dataset_adjacency_map = el_map.get_litke_array_adj_mat_by_array_id(dataset.array_id)

    if args.cell_list is not None:

        with open(args.cell_list, 'r') as cell_id_file:
            cell_id_list = list(
                map(lambda x: int(x), cell_id_file.readline().strip('\n').split(',')))

        eis_by_cell_id = {cell_id: dataset.get_ei_for_cell(cell_id).ei for cell_id in cell_id_list}
    else:
        cell_id_list = dataset.get_all_cells_of_type(args.cell_type)
        eis_by_cell_id = {cell_id: dataset.get_ei_for_cell(cell_id).ei for cell_id in cell_id_list}

    # use the initialized basis if specified, otherwise specify the number of basis vectors
    initial_basis = None
    if args.initialize_basis is not None:
        with open(args.initialize_basis, 'rb') as pfile:

            basis_dict = pickle.load(pfile)
            initial_basis = basis_dict['basis']


    shift_tuple = (-args.before, args.after)

    if initial_basis is None:
        decomposition_dict, basis_waveforms, mse = spat_decomp.spatial_cont_time_optimization(
            eis_by_cell_id,
            dataset_adjacency_map,
            args.spatial_reg,
            compute_device,
            n_basis_vectors=args.nbasis,
            snr_abs_threshold=args.thresh,
            supersample_factor=args.upsample,
            shifts=shift_tuple,
            grid_search_step=args.grid_step,
            grid_search_top_n=args.grid_top_n,
            fine_search_width=args.fine_search_width,
            grid_search_batch_size=args.grid_batch_size,
            maxiter_spatial_reg_decomp=args.maxiter,
            renormalize_data_waveforms_waveform_fit=args.renormalize,
            l1_regularize_lambda=args.weight_reg,
            sobolev_regularize_lambda=args.sobolev_reg,
        )

    else:
        decomposition_dict, basis_waveforms, mse = spat_decomp.spatial_cont_time_optimization(
            eis_by_cell_id,
            dataset_adjacency_map,
            args.spatial_reg,
            compute_device,
            initialized_basis_vectors=initial_basis,
            snr_abs_threshold=args.thresh,
            supersample_factor=args.upsample,
            shifts=shift_tuple,
            grid_search_step=args.grid_step,
            grid_search_top_n=args.grid_top_n,
            fine_search_width=args.fine_search_width,
            grid_search_batch_size=args.grid_search_batch_size,
            maxiter_spatial_reg_decomp=args.maxiter,
            renormalize_data_waveforms_waveform_fit=args.renormalize,
            l1_regularize_lambda=args.weight_reg,
            sobolev_regularize_lambda=args.sobolev_reg,
        )
        pass

    with open(args.output, 'wb') as joint_fit_file:
        metadata_dict = {
            'l1_reg': args.weight_reg,
            'sobolev_reg': args.sobolev_reg,
            'maxiter': args.maxiter,
            'padding': shift_tuple,
            'upsample': args.upsample,
            'thresh': args.thresh
        }

        pickle_dict = {
            'decomposition': decomposition_dict,
            'waveforms': basis_waveforms,
            'mse': mse
        }

        pickle.dump(metadata_dict, joint_fit_file)
        pickle.dump(pickle_dict, joint_fit_file)
