# Optimization-based EI compartment decomposition

## Description

This algorithm decomposes the EIs into shifted superpositions of basis waveforms by iteratively optimizing the basis 
waveform shape, time shifts, and amplitudes in alternating steps. The algorithm performs the optimization over an entire
population of cells (for example, the optimization works simultaneously over all sufficiently large amplitude electrodes
for cells of a specified cell type). **In its simplest form, the user specifies only the number of basis waveforms and not
the waveforms themselves (The waveforms shapes are initialized randomly and learned during the process). With sufficient 
regularization of the amplitudes, the basis waveforms tend to converge to the stereotypical axon, dendrite, and soma waveforms.**

The decomposition problem is not convex. Although the algorithm tends to produce comparable outputs
when run repeatedly on the same inputs, it is not a bad idea to fit a dataset a few times and pick the best fit.

The algorithm appears to perform reasonably for all four major cell types, as well as for the smooth monostratified cells.

#### Example learned basis waveforms (randomly initialized)

![example_basis_waveforms](example_images/example_basis_waveforms.png)

#### Example fits for several electrodes

Blue is ground truth, orange is the fit.

![example_fits](example_images/example_fits.png)

#### Decomposition of example cells into compartments

Size of circle represents relative amplitude of each compartment. Blue is somatic,
green is dendrite, black is axonic. Note that each given electrode could have axonal, dendritic, and somatic contributions.
![example_cell](example_images/example_cell.png)
![second_example_cell](example_images/second_example_cell.png)

## How to use

The simplest way to use the algorithm is to use the wrapper `fourier_nmf_decomposition.py`. The algorithm typically 
converges in about 20-25 iterations. 

#### Synopsis 
```shell script
python fourier_nmf_decomposition <path-to-dataset> <name-of-dataset> <cell-type> <output-pickle-file> [optional flags]
```

#### Dependencies
* Python >= 3.7
* numpy, scipy, pytorch >= 1.5.0
* Artificial retina software pipeline (visionloader, etc.)

#### Options
* ```-n num_basis_vectors``` Number of basis waveforms, default value 3
* ```-m max_iterations``` Maximum number of iterations to run the algorithm, default value 25
* ```-w l1_penalty``` Lambda value for L1 regularization of the amplitudes, default value of 7.5e-2 works well.
* ```-s sobolev_penalty``` Lambda value for regularizing smoothness of waveforms with low pass filter. Default value 1e-3.
Generally doesn't affect the result too much
* ```-u upsample_factor``` Upsample factor, default value 5 (bspline upsample EIs to 100 kHz)
* ```-b before_shifts``` Maximum forward shift of the basis waveforms (units upsampled samples), default value 100. Note that this also corresponds
to the amount of left zero padding for the EI
* ```-a after_samples``` Maximum backward shift of the basis waveforms (units upsampled samples), default value 100. Note that this also corresponds
to the amount of right zero padding for the EI
* ```-t ei_threshold``` Amplitude threshold for including an EI electrode in the calculation. If the amplitude exceeds this value,
it is included; if not, it is excluded. Default value 5
* ```-c cell_list``` Override the cell-type positional argument with a \n separated file of cell ids

#### Example

Decomposition of ON parasols, run on GPU3, with 7.5e-3 L1 lambda

```shell script
CUDA_VISIBLE_DEVICES=3, python fourier_nmf_decomposition.py /path/to/data/2018-03-01-4/data000 data000 "ON parasol" /path/to/output/pickle.p -w 7.5e-3
```

#### Return values

## How it works