import visionloader as vl

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate cell id list for given cell type for STA computation on a subset of cells in the dataset')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('output', type=str, help='path to save location')
    parser.add_argument('cell_type', type=str, help='relevant cell type')
    parser.add_argument('-m', '--max_n', type=int, default=-1, help='maximum number of cells to get; for DEBUG only')

    args = parser.parse_args()

    dataset = vl.load_vision_data(args.ds_path, args.ds_name, include_params=True)

    all_cells_of_type = dataset.get_all_cells_of_type(args.cell_type)
    if args.max_n != -1:
        all_cells_of_type = all_cells_of_type[:args.max_n]

    with open(args.output, 'w') as output_file:
        output_file.write(','.join(map(lambda x: str(x), all_cells_of_type)))
