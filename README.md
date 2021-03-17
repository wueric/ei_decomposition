# Optimization-based EI compartment decomposition

## Dependencies:
* artificial-retina-software-pipeline (visionloader)
* pytorch >= 1.5
* numpy, scipy
* sklearn

This runs on GPU only; I have never tried it on CPU and it would likely be intolerably slow on CPU.

## Description
This algorithm decomposes the EIs into scaled and shifted superpositions of basis waveforms by iteratively optimizing
over the (time shifts, amplitudes) and waveform shapes in alternating steps. The algorithm performs the optimization over
an entire population of cells (for example, all sufficiently large amplitude electrodes for ON parasols in a given recording
are handled at the same time). In the new and improved version, the initial waveform shapes for the optimization are calculated
by PCA clustering of the recorded waveforms or by manually selecting representative electrodes for soma, axon, and dendrite. 
This allows for better fits and more stable and repeatable results when compared with random initialization.

The algorithm appears to perform reasonably for all four major cell types. Fits for large cells need to be checked
over carefully, and amacrines are not fit well at all (EIs for amacrines are not well defined)

The best way to run the entire pipeline is to write a wrapper shell script (see example.sh). There are several data
preparation steps, and different methods for initializing the waveforms require different scripts. If L1L2 group sparsity
regularization is desired, manual intervention part way through the process will be necessary to assign the groups.

## Component scripts

### Specifying which cells to decompose

A .csv file containing a list of cell ids is necessary to specify which cells the decomposition is applied to. This .csv
has format

```
1,2,3,4,5,6,7,8,9,10
```

where each number is a cell id. All relevant cell ids are contained in a single line, and are separated by commas. 

This file can be manually generated, or can be automatically generated from an existing Vision cell type classification
using the following two scripts.

#### generate_typed_cell_id_list.py

This script generates a .csv file where the relevant cell ids are all cells that exactly match a specified
cell type from an existing Vision cell type classification. For the major cell types, you will likely want to
use this method.

##### SYNOPSIS
```shell script
python generate_typed_cell_id_list.py <Vision-ds-path> <Vision-ds-name> <path-to-output-csv> <Cell-type-name>
```

##### Example usage:
This command exports cell ids of all ON parasols in the dataset and saves it in `path/to/your/csv`
```shell script
python generate_typed_cell_id_list.py /Volumes/Analysis/9999-99-99/data000 data000 path/to/your/csv "ON parasol"
```


#### generate_typed_prefix_cell_id_list.py

This script generates a .csv file where the relevant cell ids correspond to cells whose cell type name begins with
the specified prefix (i.e. if you have "ON parasols" and "ON midgets" and you specify "ON" as the prefix, this will
grab both ON parasols and midgets). This is useful for wrangling with more complicated cell type classifications for large
cells, where the relevant cells may be in different subcategories.

##### SYNOPSIS
```shell script
python generate_typed_prefix_cell_id_list.py <Vision-ds-path> <Vision-ds-name> <path-to-output-csv> <Cell-type-prefix>
```


##### Example usage:
This command exports cell ids of all cells whose type name begins with "OFF SM combine" in the dataset and saves it in `path/to/your/csv`
```shell script
python generate_typed_prefix_cell_id_list.py /Volumes/Analysis/9999-99-99/data000 data000 path/to/your/csv "OFF SM combine"
```

### Extracting the relevant data from Vision

#### create_data_pickle.py

This command extracts the EIs for the relevant cells from Vision and saves the result as a pickle file. The output
of this script is the data source for the main optimization routine.

##### SYNOPSIS
```shell script
python create_data_pickle.py <Vision-ds-path> <Vision-ds-name> <path-to-input-csv> <path-to-save-data pickle>
```

##### Example usage 

This extracts and saves the EIs for the cell ids specified in `path/to/your/csv` into a pickle file located at `path/to/save/data_pickle.p`

```shell script
python create_data_pickle.py /Volumes/Analysis/9999-99-99/data000 data000 path/to/your/csv path/to/save/data_pickle.p
```

### Initializing waveforms (optional but strongly recommended)

There are two ways to generate initial waveform shapes for the compartments before the main optimization (You can also
learn the waveform shapes de novo during the main optimization, but this generally gives crummier and less repeatable 
results). These methods are (1) time-shift alignment of waveforms followed by PCA clustering; and (2) Manual selection of
representative compartment waveforms.

#### pca_init_basis_waveforms.py

Time-shift alignment of waveforms followed by PCA clustering: this method works by temporally shifting each waveform
such that the maximum deviation from zero occurs at the same time (i.e. aligning the peaks), renormalizing all of the wavforms
such that they have L2 norm 1, performing PCA and EM clustering with 3 cluster centers, and then reading off the cluster means
as the initial basis waveforms.

This method works really well for the four major cell types, where the compartment waveforms are reasonably well defined
and temporal alignment is easy. This works less well for the rare cell types, probably because temporal alignment of the peaks
is very tricky when most electrodes contain a superposition of basis waveforms.

##### SYNOPSIS
```shell script
python pca_init_basis_waveforms.py <input-data-pickle> <output-basis-pickle> [optional-one-letter-args]
```

##### Optional arguments
* --upsample, -u, integer value, default 5. Upsample factor. For example, 5 corresponds to bspline interpolation + 5x upsampling. This parameter
is shared with the main optimization, and  setting the parameter here corresponds to setting it for the main optimization
* --before, -b, integer value, default 100. Number of upsampled samples to zero pad the front of the EI. Also corresponds to the lower
bound for the range of shifts that the main optimization considers when doing temporal alignment. This parameter
is shared with the main optimization, and  setting the parameter here corresponds to setting it for the main optimization
* --after, -a, integer value, default 100. Number of upsampled samples to zero pad the back of the EI. Also corresponds to the lower
bound for the range of shifts that the main optimization considers when doing temporal alignment. This parameter
is shared with the main optimization, and  setting the parameter here corresponds to setting it for the main optimization
* --thresh, -t, float value, default 5.0. Mininum deviation from zero required to include an electrode in the calculation. This parameter
is shared with the main optimization, and  setting the parameter here corresponds to setting it for the main optimization
* --alignment_sample, -l, integer value, default 200. Sample number to align the peak to. **Adjust this value if the waveforms
are cut off in time**
* --nbasis, -n, integer value, default 3. Number of basis waveforms (cluster centers). Should be 3, unless you have some other
decomposition in mind that isn't (soma, dendrite, axon)
* --n_pca_components, -p, integer value, default 5. Number of dimensions for the PCA.

##### Example usage

This does clustering for all of the cells in `input/data_pickle.p`, and upsamples 5x, and pads 100 zero samples at the front
and 200 zero samples at the back of each waveform, and saves the resulting waveforms to `output/basis_pickle.p`

```shell script
python pca_init_basis_waveforms.py input/data_pickle.p output/basis_pickle.p -u 5 -b 100 -a 200
```

#### save_initialized_waveforms_basis.py

This allows the user to manually select basis waveforms. For rare cell types, this is likely what you will want to use.

##### SYNOPSIS
```shell script
python save_initialized_waveforms_basis.py <input-data-picle> <output-basis-pickle> -c <selected-cell-id> -e <selected electrode sequence> [optional-one-letter-args]
```

##### Mandatory flag arguments
* --cell_id, -c, integer value. The cell id of the cell that you want to take basis waveforms from
* --electrodes, -e, sequence of integers. The indices of the electrodes corresponding to the basis waveforms. Zero-indexed.

##### Optional arguments
* --upsample, -u, integer value, default 5. Upsample factor. For example, 5 corresponds to bspline interpolation + 5x upsampling. This parameter
is shared with the main optimization, and  setting the parameter here corresponds to setting it for the main optimization
* --before, -b, integer value, default 100. Number of upsampled samples to zero pad the front of the EI. Also corresponds to the lower
bound for the range of shifts that the main optimization considers when doing temporal alignment. This parameter
is shared with the main optimization, and  setting the parameter here corresponds to setting it for the main optimization
* --after, -a, integer value, default 100. Number of upsampled samples to zero pad the back of the EI. Also corresponds to the lower
bound for the range of shifts that the main optimization considers when doing temporal alignment. This parameter
is shared with the main optimization, and  setting the parameter here corresponds to setting it for the main optimization
* --thresh, -t, float value, default 5.0. Mininum deviation from zero required to include an electrode in the calculation. This parameter
is shared with the main optimization, and  setting the parameter here corresponds to setting it for the main optimization
* --alignment_sample, -l, integer value, default 200. Sample number to align the peak to. **Adjust this value if the waveforms
are cut off in time**

##### Example usage

This grabs data from electrodes 10, 11, and 12 from cell 256, upsamples 5x, and pads 100 zeros at the front and 200 zeros
at the back of every waveform, and saves the resulting waveforms to `output/basis_pickle.p`

```shell script
python save_initialized_waveforms_basis.py input/data_pickle.p output/basis_pickle.p -c 256 -e 10 11 12 -b 100 -a 200 -u 5
```

#### Inspecting the output of the initialization step




# DEPRECATED (EVERYTHING BELOW THIS CORRESPONDS TO THE OLD THREE STAGE ALGORITHM)

### Description

This algorithm decomposes the EIs into scaled and shifted superpositions of basis waveforms by iteratively optimizing the basis 
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

The return values are stored in a Python pickle file. The pickle file contains two distinct dictionaries. These dictionaries
must be loaded with consecutive calls of ```pickle.load()```.
1. The first dictionary is the optimization parameter dictionary. It contains all of the optimization hyperparameters for
record-keeping.
2. The second dictionary contains the results of the optimization. The keys in this dictionary are:
    * ```'mse'```: -> ```float``` the final MSE value over all waveforms at the end of the optimization
    * ```'waveforms``` -> ```np.ndarray``` the basis waveforms found by the optimization. Has shape ```(n_basis_waveforms, n_timepoints)```
    Note that because the waveforms are fit from random initialization, the compartments will occur in random order.
    * ```'decomposition'``` -> ```Dict[int, Tuple[np.ndarray, np.ndarray]]``` Dict mapping cell id integer to the decompositions.
    The first ```np.ndarray``` contains the amplitudes, and has shape ```(n_electrodes, n_basis_waveforms)```, with the order of
    the columns corresponding to the order of basis waveforms in ```waveforms```. The second ```np.ndarray``` contains the shifts,
    in units of supersampled samples, and has shape ```(n_electrodes, n_basis_waveforms)```. The order of the columns corresponds
    to the order of the basis waveforms in ```waveforms```. Negative is forward shift, and positive is a delay.

## How it works

The algorithm has the following steps

1. Randomly initialize basis waveforms, shifts, and amplitudes. Basis waveforms can take any value, shifts must be within the range
specified by the user, and amplitudes must be nonnegative.
2. Iteratively fit by alternating between the following three steps
    1. Given fixed basis waveforms and time shifts, solve for the scaling amplitudes on each basis waveform. This is formulated 
    as nonnegative least squares (the amplitudes must be nonnegative), with optional L1 regularization to promote sparsity
    among the amplitudes (electrodes often tend be just one compartment, for example, pure axonic electrodes). The problem is separable
    over each (cell, electrode) pair, and it is solved in parallel for each.
    2. Given fixed amplitudes and time shifts, solve for the basis waveforms in Fourier domain. Because this problem is solved
    in Fourier domain (convenient for dealing with the time shifts, because in Fourier domain the shifts become elementwise
    multiplication), the problem is formulated as complex linear least squares, with an extra factor to account for the shifting.
    The problem is separable over each frequency, and is solved in parallel for each.
    3. Given fixed amplitudes and waveforms, solve for the timeshifts with greedy deconvolution.
    
There is a document with mathematical details... The details are quite unpleasant...