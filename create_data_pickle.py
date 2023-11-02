import argparse
import pickle

import visionloader as vl
from lib.util_fns import bspline_upsample_waveforms, parse_ignore_electrodes

DESCRIPTION = "Extracts EIs from Vision and puts them into a pickle file, one entry for every cell. Not for publication; " + \
              "this requires Vision datasets and machinery that the outside world does not have"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('cell_list_file', type=str, help='path to cell list file (comma separated integers of cell id')
    parser.add_argument('output', type=str, help='path to output pickle file')
    parser.add_argument('--upsample', '-u', type=int, default=2, help='upsample factor')
    parser.add_argument('--ignore_el', '-i', type=str, default=None,
                        help='path to CSV containing indices of electrodes to ignore')
    args = parser.parse_args()

    els_to_ignore = None
    if args.ignore_el is not None:
        with open(args.ignore_el, 'r') as ignore_el_file:
            line = ignore_el_file.readline()
            els_to_ignore = parse_ignore_electrodes(line)

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
    if els_to_ignore is not None:
        for cell_id, orig_ei in eis_by_cell_id.items():
            orig_ei[els_to_ignore, :] = 0.0

    upsampled_eis_by_cell_id = {cell_id : bspline_upsample_waveforms(raw_eis, args.upsample)
                                for cell_id, raw_eis in eis_by_cell_id.items()}

    #padded_upsampled_eis_by_cell_id = {cell_id : np.pad(upsampled_ei, [(0, 0), (args.before, args.after)])
    #                                   for cell_id, upsampled_ei in upsampled_eis_by_cell_id.items()}

    with open(args.output, 'wb') as pfile:
        data_dict = {
            'eis_by_cell_id': upsampled_eis_by_cell_id,
            'electrode_map': dataset_el_map,
            'upsample': args.upsample,
        }

        if els_to_ignore is not None:
            data_dict['ignore_el'] = els_to_ignore

        pickle.dump(data_dict, pfile)
