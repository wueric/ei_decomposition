import visionloader as vl

import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Select initial basis waveforms')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('cell_type', type=str, help='cell type of interest')
    parser.add_argument('basis_pickle', type=str, help='path to input pickle file')
    parser.add_argument('--cell_id', '-c', type=int, help='cell id')
    parser.add_argument('--electrodes', '-e', type=int, nargs='+', help='electrodes')

    args = parser.parse_args()

    print("Loading data")
    dataset = vl.load_vision_data(args.ds_path,
                                  args.ds_name,
                                  include_params=True,
                                  include_ei=True)

    ei_for_template = dataset.get_ei_for_cell(args.cell_id).ei
    basis_waveforms = ei_for_template[list(args.electrodes),:]

    with open(args.basis_pickle) as pfile:
        pickle_dict = {
            'basis' : basis_waveforms
        }
        pickle.dump(pickle_dict, pfile)

