import visionloader as vl

import torch

import pickle

import lib.ei_decomposition as ei_decomp

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Refit EI decomposition with known basis waveforms and shifts')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('cell_type', type=str, help='cell type of interest')
    parser.add_argument('input_pickle', type=str, help='path to input pickle file')
    parser.add_argument('output_pickle', type=str, help='path to output pickle file')

    args = parser.parse_args()

    dataset = vl.load_vision_data(args.ds_path,
                                  args.ds_name,
                                  include_params=True,
                                  include_ei=True)

    print("Loading data")
    with open(args.input_pickle, 'rb') as pfile:
        metadata = pickle.load(pfile)
        decomposition_dict = pickle.load(pfile)

