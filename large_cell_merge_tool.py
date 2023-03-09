import argparse
import os
from typing import List

import spikebinning

import visionloader as vl
import visionwriter as vw


def parse_multiunit_file(lines) -> List[List[int]]:

    multi_units = []
    for line in lines:
        split_units = list(map(int, line.strip('\n').split(',')))
        multi_units.append(split_units)
    return multi_units


DESCRIPTION = "Tool for merging large cell EIs by combining .neuron files and using Vision to recompute EIs. NOT FOR PUBLICATION"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('ds_path', type=str, help='path to original Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of original Vision dataset')
    parser.add_argument('cell_list_file', type=str, help='path to multi-unit oversplits to merge')
    parser.add_argument('output_ds_path', type=str, help='path to output dataset, create if not exist')
    parser.add_argument('output_ds_name', type=str, help='name of output dataset')
    parser.add_argument('output_cell_id_path', type=str, help='path to output cell id file')

    args = parser.parse_args()

    with open(args.cell_list_file, 'r') as multiunit_file:
        lines = multiunit_file.readlines()
    multiunit_list = parse_multiunit_file(lines)

    reference_dataset = vl.load_vision_data(args.ds_path,
                                            args.ds_name,
                                            include_neurons=True)

    merged_spike_times = {}
    for multiunit in multiunit_list:
        spike_times_multiunit = [reference_dataset.get_spike_times_for_cell(cell_id) for cell_id in multiunit]
        merged_spike_times[multiunit[0]] = spikebinning.merge_multiple_sorted_array(spike_times_multiunit)

    os.makedirs(args.output_ds_path, exist_ok=True)
    with vw.NeuronsFileWriter(args.output_ds_path, args.output_ds_name) as nfw:
        nfw.write_neuron_file(merged_spike_times,
                              reference_dataset.get_ttl_times(),
                              reference_dataset.n_samples)

    with open(args.output_cell_id_path, 'w') as output_id_file:
        output_id_file.write(','.join(list(map(lambda x: str(x), list(merged_spike_times.keys())))))
