import argparse
import os
from typing import List, Dict
from dataclasses import dataclass

import numpy as np

import spikebinning

import visionloader as vl
import visionwriter as vw


@dataclass
class RetimeUnit:
    units: List[int]
    retime_electrode: int


def parse_multiunit_file_with_alignment_electrode(lines) -> List[RetimeUnit]:
    multi_units = []
    for line in lines:
        units, electrode = line.strip('\n').split(';')
        electrode = int(electrode)
        split_units = list(map(int, units.split(',')))
        multi_units.append(RetimeUnit(split_units, electrode))
    return multi_units


def retime_merge_spikes_by_ei(vision_dataset: vl.VisionCellDataTable,
                              retime_unit: RetimeUnit) -> np.ndarray:

    alignment_point_by_unit = {}
    for cell_id in retime_unit.units:
        ei = vision_dataset.get_ei_for_cell(cell_id).ei
        ei_electrode_channel = ei[retime_unit.retime_electrode, :]
        alignment_point = np.argmax(np.abs(ei_electrode_channel))
        alignment_point_by_unit[cell_id] = alignment_point

    median_timepoint = int(np.median([val for val in alignment_point_by_unit.values()]))
    spike_times_to_merge = [] # type: List[np.ndarray]
    for cell_id in retime_unit.units:
        spike_times = vision_dataset.get_spike_times_for_cell(cell_id)
        shift_amount = median_timepoint - alignment_point_by_unit[cell_id]

        shifted_spike_times = spike_times + shift_amount
        spike_times_to_merge.append(shifted_spike_times)

    return spikebinning.merge_multiple_sorted_array(spike_times_to_merge)


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
    multiunit_list = parse_multiunit_file_with_alignment_electrode(lines)

    reference_dataset = vl.load_vision_data(args.ds_path,
                                            args.ds_name,
                                            include_neurons=True,
                                            include_ei=True)

    merged_spike_times = {}
    for multiunit in multiunit_list:
        merged_spike_times[multiunit.units[0]] = retime_merge_spikes_by_ei(reference_dataset,
                                                                           multiunit)

    os.makedirs(args.output_ds_path, exist_ok=True)
    with vw.NeuronsFileWriter(args.output_ds_path, args.output_ds_name) as nfw:
        nfw.write_neuron_file(merged_spike_times,
                              reference_dataset.get_ttl_times(),
                              reference_dataset.n_samples)

    with open(args.output_cell_id_path, 'w') as output_id_file:
        output_id_file.write(','.join(list(map(lambda x: str(x), list(merged_spike_times.keys())))))

