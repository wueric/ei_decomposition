import argparse
import numpy as np

import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Assign groups to the initialized basis waveforms')
    parser.add_argument('basis_pickle', type=str, help='path to output pickle file')
    parser.add_argument('groups_csv', type=str, help='path to groups assignment csv')

    args = parser.parse_args()

    with open(args.basis_pickle, 'rb') as pfile:
        basis_vectors_dict = pickle.load(pfile)

    groups_list = []
    with open(args.groups_csv, 'r') as groups_file:
        contents = groups_file.read().strip('\n')

        lines = contents.split('\n')
        for line in lines:
            groups_list.append(np.array(map(lambda x: int(x), line.split(','))))

    basis_vectors_dict['group_assignments'] = groups_list

    with open(args.basis_pickle, 'wb') as pfile:
        pickle.dump(basis_vectors_dict, pfile)
