# Optimization-based EI compartment decomposition

## Description

This repository contains the code for fitting the EI decomposition, from the paper "Decomposition of retinal ganglion
cell electrical images for cell type and functional inference".

The EI decomposition algorithm decomposes EIs into scaled and shifted superpositions of learned basis waveforms by
iteratively optimizing
over (time shifts and amplitudes for each electrode) and waveform shape. The algorithm is run separately (but in
parallel) for each cell, meaning
that a separate set of basis waveforms are learned for each individual cell. This enables comparisons between the
learned
waveforms returned by the decomposition algorithm across cells of different type, as the algorithm makes no assumptions
about cell type.

Implementation details for the algorithm can be found in the Supplemental Information associated with the paper.

## Dependencies:

* pytorch >= 1.5
* numpy, scipy
* sklearn

The software here has been tested only on NVIDIA GPUs. In principle it should work with the CPU-only version of Pytorch,
but this has not been tested, and CPU-only runtime is expected to be exceedingly poor.

## High-level description of contents

#### Main scripts (for everyone)

* `pca_init_basis_waveforms.py`: This script initializes the waveform shape prior means using PCA clustering. Using this
  script is strictly optional, as the waveform means do not have to be initialized using this script.
* `assign_basis_groups.py`: This script modifies the initialized basis waveforms pickle file to include the group
  assignments for the L21 group-sparsity penalty. Running this script is required if the L21 group-sparsity regularizer
  is used.
* `fourier_nmf_single_cell_with_gaussian_prior.py`: This is the main EI decomposition fitting script. It performs the
  iterative alternating optimization for learning basis waveforms, amplitudes, and time shifts for every cell. This is
  the only script that runs on GPU.

Note that external (non-Chichilnisky lab) users will have to save their data in a Python .pickle file themselves. The
specific format for that file will be described below.

#### Internal scripts (for Chichilnisky lab users only)

* `generate_typed_cell_id_list.py`: Generates a .txt file containing the cell ids of all cells with the specified cell
  type in Vision. This is useful for performing cell-type specific analyses using the decomposition.
* `create_data_pickle.py`: Exports the EIs from Vision for the cell ids specified by a .txt file, and saves it in the
  appropriately-formatted Python .pickle file.

## External Workflow (non-Chichilnisky lab users)

This is meant to be a quick summary of how to run the EI decomposition code.
Descriptions of the command line arguments for all scripts is provided at the bottom of this README.

#### Step 1: exporting your data

The data .pickle file contains a single pickled Python Dict. The required keys and corresponding values are:

1. `eis_by_cell_id`: A nested Python Dict, with integer-valued keys for cell ids, and EIs stored as a (num_electrodes,
   num_time_samples) floating-point np.ndarray. Note that the temporal resolution of the time shifts fitted by the
   decomposition is determined by sampling rate of the EIs saved here, and if you want increased temporal resolution you
   will need to upsample the EIs yourself prior to saving this file.
2. `upsample_factor`: The temporal upsampling factor. The value here is not explicitly used by the decomposition fitting
   procedure, but is saved with the computed decomposition outputs for record-keeping.
3. `ignore_el`: np.ndarray of non-negative integer indices, corresponding to the 0-indexed indices of recording
   electrodes that should be ignored by the decomposition. In general, it is reasonable to ignore shorted or broken
   recording electrodes.

#### Step 2: Initialize the waveform shape prior means

There are two ways to initialize the waveform shape prior. The first option is to run`pca_init_basis_waveforms.py`,
which uses PCA and Gaussian mixture model clustering to identify unique waveform shapes that appear often, and
initializes the means as the
means of each cluster.

Alternatively, you can specify your own basis means. This requires generating and pickling a Python dict. The required
keys and corresponding values are:

1. `basis`: the cluster centers, a np.ndarray with shape (num_basis, num_time_samples). Note that the number of time
   samples here must be the same as in the saved data pickle.
2. `alignment_sample`: integer, the sample number to temporally align. The decomposition fitting procedure aligns the
   basis waveforms such that their absolute maximum value occurs at this position, to prevent the basis waveforms from
   being cut off. It is typically a good idea to set this value to somewhere in the middle.
3. `upsample`: the upsample factor. This is not used, but is saved with the output for record-keeping.

#### Step 3: Specify the L21 group sparsity groups

Run `assign_basis_groups.py` using the basis pickle file saved in step 2.

#### Step 4: Fit the decomposition

Run `fourier_nmf_single_cell_with_gaussian_prior.py` to perform the decomposition.

## Internal Workflow (Chichilnisky lab users)

The workflow using Vision data is

#### Step 1. Compile a list of cells to analyze

Save these cell ids in a single-line comma-delimited file. You can use `generate_typed_cell_id_list.py` as a helper
script if you like.

#### Step 2. Extract the data from Vision

Run `create_data_pickle.py` to export the EIs for the cells that you want to analyze. If desired, temporal upsampling
needs to be specified here.

#### Step 3. Initialize the basis waveform shape prior means

(see above). Alternatively, you can use the provided standard prior.

#### Step 4. Assign basis groups

Run `assign_basis_groups.py`

#### Step 5. Fit the decomposition

Run `fourier_nmf_single_cell_with_gaussian_prior.py`

## Script arguments

### Specifying which cells to decompose (internal use only)

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

### Extracting the relevant data from Vision (internal use only)

#### create_data_pickle.py

This command extracts the EIs for the relevant cells from Vision and saves the result as a pickle file. The output
of this script is the data source for the main optimization routine.

##### SYNOPSIS

```shell script
python create_data_pickle.py <Vision-ds-path> <Vision-ds-name> <path-to-input-csv> <path-to-save-data pickle>
```

##### Example usage

This extracts and saves the EIs for the cell ids specified in `path/to/your/csv` into a pickle file located
at `path/to/save/data_pickle.p`

```shell script
python create_data_pickle.py /Volumes/Analysis/9999-99-99/data000 data000 path/to/your/csv path/to/save/data_pickle.p
```

### Creating your own waveform prior mean and sparsity groups

(Optional, but if you have EI lengths that have a different number of samples you will need to do this)

#### pca_init_basis_waveforms.py

Time-shift alignment of waveforms followed by PCA clustering: this method works by temporally shifting each waveform
such that the maximum deviation from zero occurs at the same time (i.e. aligning the peaks), renormalizing all of the
wavforms
such that they have L2 norm 1, performing PCA and EM clustering with 3 cluster centers, and then reading off the cluster
means
as the initial basis waveforms.

This method works really well for the four major cell types, where the compartment waveforms are reasonably well defined
and temporal alignment is easy. This works less well for the rare cell types, probably because temporal alignment of the
peaks
is very tricky when most electrodes contain a superposition of basis waveforms.

##### SYNOPSIS

```shell script
python pca_init_basis_waveforms.py <input-data-pickle> <output-basis-pickle> [optional-one-letter-args]
```

##### Optional arguments

* --upsample, -u, integer value, default 5. Upsample factor. For example, 5 corresponds to bspline interpolation + 5x
  upsampling. This parameter
  is shared with the main optimization, and setting the parameter here corresponds to setting it for the main
  optimization
* --thresh, -t, float value, default 5.0. Mininum deviation from zero required to include an electrode in the
  calculation. This parameter
  is shared with the main optimization, and setting the parameter here corresponds to setting it for the main
  optimization
* --alignment_sample, -l, integer value, default 200. Sample number to align the peak to. **Adjust this value if the
  waveforms
  are cut off in time**
* --nbasis, -n, integer value, default 3. Number of basis waveforms (cluster centers). Should be 3, unless you have some
  other
  decomposition in mind that isn't (soma, dendrite, axon)
* --n_pca_components, -p, integer value, default 5. Number of dimensions for the PCA.

##### Example usage

This does clustering for all of the cells in `input/data_pickle.p`, and upsamples 5x, and pads 100 zero samples at the
front
and 200 zero samples at the back of each waveform, and saves the resulting waveforms to `output/basis_pickle.p`

```shell script
python pca_init_basis_waveforms.py input/data_pickle.p output/basis_pickle.p -u 5 -b 100 -a 200
```

### Saving groups for group sparsity

If you are using the group sparsity L12 penalty, in addition to the basis prior means, you have to manually specify the
groups.
Since the PCA clustering method outputs basis vectors in random order, manual inspection and annotation is necessary.

Once you have you have decided the groups, use the following script to bind the groups to the basis prior means:

#### assign_basis_groups.py

The input csv each group on a different line, and the indices within each group separated by a comma. For example,

```shell script
0
1,2
```

corresponds to two groups. The first group contains basis waveform 0 alone, and the second group contains basis
waveforms
1 and 2 together.

##### SYNOPSIS

```shell script
python assign_basis_groups.py <basis-pickle-path> <group-csv-path>
```

##### Example usage

This updates an already existing basis pickle at `input/basis_pickle.p` with the group assignments found
in `input/group_assignment.csv`.

```shell script
python assign_basis_groups.py input/basis_pickle.p input/group_assignment.csv
```

## Running the main optimization

The main optimization script is called `fourier_nmf_single_cell_with_gaussian_prior.py`.

#### fourier_nmf_single_cell_with_gaussian_prior.py

##### SYNOPSIS

```shell script
python fourier_nmf_single_cell_with_gaussian_prior.py <ei_data_pickle> <basis_prior_pickle> <output> [optional flags]
```

##### Optional arguments

* --maxiter, -m, integer value, default value 3. Number of total alternating optimization iterations (each consisting of
  one amplitude + shifts step AND one waveform fitting step) to run. 3 is a reasonable value, more iterations will lead
  to
  marginally better fits.
* --weight_reg, -w, float, default 1e-1. Weight to place on the group sparsity prior for the amplitudes + shifts
  optimization.
* -b, --before, positive integer value. Amount of pad/shifts forwards in time used in the shifts search component of the
  amplitudes optimization. Determines the search space for the shifts search. In units of upsampled samples
* -a, --after, positive integer value. Amount of pad/shifts backwards in time used in the shifts search component of the
  amplitudes optimization. Determines the search space for the shifts search. In units of upsampled samples
* --grid_step, positive integer value, default value 5. Width of the steps in the coarse grid search over time during
  the amplitudes + shifts
  optimization. Larger value runs faster, but may result in suboptimal time shifts
* --grid_top_n, positive integer value, default 4. Number of top candidates from the coarse time grid search to expand
  upon
  for the fine search. Larger value runs slower, but may result in better time shifts.
* --fine_search_width, positive integer value, default 2. Width (on either side) of the fine time search. For each of
  the
  top n points from the coarse search, expand the neighborhood of shifts ranging from -fine_search_width to
  +fine_search_width
* --grid_batch_size, positive integer value, default 8192. Grid search batch size. Default value appears to be
  reasonable
  and runs on GPUs with 12 Gb VRAM without issue. Increasing this number does not appear to drastically improve
  performance.
* --thresh, -t, float, default value 5.0. EI amplitude cutoff, below which the electrode is not used in the optimization
* --renormalize_loss, -r, store True, default False. Renormalizes the terms of the MSE loss during the waveform shape
  optimization
  so that each electrode contributes approximately equally to the objective, even if the overall magnitudes of the
  recorded signals
  are different. Leaving this False has the tendency to let a few large-amplitude channels dominate the waveform shape
  optimization,
  and hence ignore the lower-amplitude channels entirely. Setting this to True has the tendency to increase the amount
  of noise
  in the basis waveforms, as low-amplitude (noisier) channels are given a higher weight. All of the analyses in the
  paper set this
  flag to True.
* --renormalize_penalty, -p, store True, default False. Renormalizes the weight on the (group) sparsity penalty so that
  the sparsity
  is about the same for each channel (i.e. each channel should be about equally sparse; there should be no amplitude
  dependence). All
  of the analyses in the paper set this flag to True.
* --group, -g, store True, default False. Use group sparsity regularization. If not set, then default to L1
  regularization.
* --prior_weight, -pw, float, default value 1.0. Weight placed on the waveform shape Gaussian prior during the waveform
  shape
  optimization. Smaller values will allow the waveforms to vary more from the prior means, while larger values tie the
  waveforms
  more closely to the prior mean
* --prior_width, -psigma, float, default value 5.0. Width (in units of upsampled samples) of the Gaussian prior
  covariance. By
  default we use an RBF kernel to construct the covariance matrix. The exact form of the covariance matrix can be
  changed in the source code.
* eps_cutoff, -e, float, default 1e-3. Cutoff for the FISTA amplitudes + shifts solver.
* --opt_iter, -o, positive integer default 10, maximum number of inner iterations fro the FISTA solver used to do the
  amplitudes + shifts
  optimization. That optimization converges fairly fast, so 10 iterations is reasonable. More iterations will produce
  marginally better fits
  but will be slower.

## Outputs

The output values are stored in a Python pickle file. The pickle file contains two distinct dictionaries. These
dictionaries
must be loaded with consecutive calls of ```pickle.load()```.

1. The first dictionary is the optimization parameter dictionary. It contains all of the optimization hyperparameters
   for
   record-keeping.
2. The second dictionary contains the results of the optimization. The keys in this dictionary are:
    * ```'mse'```: -> ```Dict``` the final MSE values over all waveforms at the end of the optimization
    * ```'waveforms``` -> ```np.ndarray``` the basis waveforms found by the optimization. Has
      shape ```(n_basis_waveforms, n_timepoints)```
      Note that because the waveforms are fit from random initialization, the compartments will occur in random order.
    * ```'decomposition'``` -> ```Dict[int, Tuple[np.ndarray, np.ndarray]]``` Dict mapping cell id integer to the
      decompositions.
      The first ```np.ndarray``` contains the amplitudes, and has shape ```(n_electrodes, n_basis_waveforms)```, with
      the order of
      the columns corresponding to the order of basis waveforms in ```waveforms```. The second ```np.ndarray``` contains
      the shifts,
      in units of supersampled samples, and has shape ```(n_electrodes, n_basis_waveforms)```. The order of the columns
      corresponds
      to the order of the basis waveforms in ```waveforms```. Negative is forward shift, and positive is a delay.
