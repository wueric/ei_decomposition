import argparse
import pickle

import visionloader as vl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='jointly decompose noramlized EIs for a given cell type into constituent waveforms by jointly optimizing waveforms, shifts, and amplitudes')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('cell_list_file', type=str, help='path to cell list file (comma separated integers of cell id')
    parser.add_argument('output', type=str, help='path to output pickle file')

    args = parser.parse_args()

    print("Loading data")
    dataset = vl.load_vision_data(args.ds_path,
                                  args.ds_name,
                                  include_params=True,
                                  include_ei=True)
    dataset_el_map = dataset.get_electrode_map()

    with open(args.cell_list_file, 'r') as cell_id_file:
        cell_id_list = list(
            map(lambda x: int(x), cell_id_file.readline().strip('\n').split(',')))

    eis_by_cell_id = {cell_id: dataset.get_ei_for_cell(cell_id).ei for cell_id in cell_id_list}

    with open(args.output, 'wb') as pfile:
        pickle.dump({
            'eis_by_cell_id' : eis_by_cell_id,
            'electrode_map' : dataset_el_map
        }, pfile)







