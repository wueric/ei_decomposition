import argparse
import numpy as np

import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Assign groups to the initialized basis waveforms')
    parser.add_argument('basis_pickle', type=str, help='path to output pickle file')
    parser.add_argument('componentwise_weights_csv', type=str, help='path to L1 component weights csv')

    args = parser.parse_args()
    with open(args.basis_pickle, 'rb') as pfile:
        basis_vectors_dict = pickle.load(pfile)
        
    componentwise_weights_list = []
    with open(args.componentwise_weights_csv, 'r') as weights_file:
        contents = weights_file.read().strip('\n')
        lines = contents.split('\n')
        
        for line in lines:
            componentwise_weights_list.append(float(line.split(':')[1]))
            
    basis_vectors_dict['componentwise_weights'] = np.array(componentwise_weights_list)
    
    with open(args.basis_pickle, 'wb') as pfile:
        pickle.dump(basis_vectors_dict, pfile)
