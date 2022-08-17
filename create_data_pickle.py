import argparse
import pickle

import visionloader as vl
from lib.util_fns import bspline_upsample_waveforms

import numpy as np

DESCRIPTION = "Extracts EIs from Vision and puts them into a pickle file, one entry for every cell. Not for publication; " + \
              "this requires Vision datasets and machinery that the outside world does not have"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('cell_list_file', type=str, help='path to cell list file (comma separated integers of cell id')
    parser.add_argument('output', type=str, help='path to output pickle file')
    parser.add_argument('--upsample', '-u', type=int, default=2, help='upsample factor')
    parser.add_argument('--before', '-b', type=int, default=40, help='left pad samples')
    parser.add_argument('--after', '-a', type=int, default=80, help='right pad samples')

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

    upsampled_eis_by_cell_id = {cell_id : bspline_upsample_waveforms(raw_eis, args.upsample)
                                for cell_id, raw_eis in eis_by_cell_id.items()}

    padded_upsampled_eis_by_cell_id = {cell_id : np.pad(upsampled_ei, [(0, 0), (args.before, args.after)])
                                       for cell_id, upsampled_ei in upsampled_eis_by_cell_id.items()}

    with open(args.output, 'wb') as pfile:
        pickle.dump({
            'eis_by_cell_id' : padded_upsampled_eis_by_cell_id,
            'electrode_map' : dataset_el_map,
            'upsample' : args.upsample,
            'before' : args.before,
            'after' : args.after
        }, pfile)







