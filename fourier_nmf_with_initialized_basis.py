import visionloader as vl

import torch

import pickle

import lib.ei_decomposition as ei_decomp

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fit EI decomposition with manually initialized basis waveforms')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('cell_type', type=str, help='cell type of interest')
    parser.add_argument('basis_pickle', type=str, help='path to input pickle file')
    parser.add_argument('output_pickle', type=str, help='path to output pickle file')
    parser.add_argument('--maxiter', '-m', type=int, default=50, help='maximum number of iterations to run')
    parser.add_argument('--weight_reg', '-w', type=float, default=1e-3, help='L1 regularization lambda for amplitudes')
    parser.add_argument('--sobolev_reg', '-s', type=float, default=1e-3,
                        help='L2 regularization for waveform derivatives')
    parser.add_argument('--upsample', '-u', type=int, default=4, help='upsample factor')
    parser.add_argument('--before', '-b', type=int, default=100, help='left shift samples')
    parser.add_argument('--after', '-a', type=int, default=100, help='right shift samples')
    parser.add_argument('--thresh', '-t', type=float, default=5.0, help='EI amplitude cutoff')

    args = parser.parse_args()

    print("Loading Vision dataset")
    dataset = vl.load_vision_data(args.ds_path,
                                  args.ds_name,
                                  include_params=True,
                                  include_ei=True)

    print("Loading initial bases")
    with open(args.basis_pickle, 'rb') as pfile:
        basis_dict = pickle.load(pfile)
        basis_vectors = basis_dict['basis']

    compute_device = torch.device('cuda')

    example_cells = dataset.get_all_cells_of_type(args.cell_type)
    eis_by_cell_id = {cell_id: dataset.get_ei_for_cell(cell_id).ei for cell_id in example_cells}

    shift_tuple = (-args.before, args.after)

    # 5e-3 was good
    decomposition_dict, basis_waveforms, mse = ei_decomp.decompose_cells_by_fitted_compartment(eis_by_cell_id,
                                                                                               compute_device,
                                                                                               initialized_basis_vectors=basis_vectors,
                                                                                               maxiter_decomp=args.maxiter,
                                                                                               l1_regularize_lambda=args.weight_reg,
                                                                                               sobolev_regularize_lambda=args.sobolev_reg,
                                                                                               renormalize_data_waveforms=True,
                                                                                               output_debug_dict=False,
                                                                                               shifts=shift_tuple,
                                                                                               supersample_factor=args.upsample,
                                                                                               snr_abs_threshold=args.thresh)

    with open(args.output_pickle, 'wb') as joint_fit_file:
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
            'mse' : mse
        }

        pickle.dump(metadata_dict, joint_fit_file)
        pickle.dump(pickle_dict, joint_fit_file)
