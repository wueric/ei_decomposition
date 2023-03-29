# Optimization-based EI compartment decomposition

## Description

This algorithm decomposes EIs into scaled and shifted superpositions of learned basis waveforms by iteratively optimizing
over (time shifts and amplitudes for each electrode) and waveform shape. The algorithm is run separately (but in parallel) for each cell, meaning
that a separate set of basis waveforms are learned for each individual cell. This enables comparisons between the learned
waveforms returned by the decomposition algorithm across cells of different type, as the algorithm makes no assumptions
about cell type.


#### Dependencies:
* artificial-retina-software-pipeline (visionloader)
* pytorch >= 1.5
* numpy, scipy
* sklearn

This runs on GPU only.

#### Workflow:

The best way to run the entire pipeline is to write a wrapper shell script (see example.sh).

The workflow (if using Vision data format) is
1. Compile a list of cell ids to analyze. Save these cell ids in a single-line comma-delimited file.
2. Run `create_data_pickle.py` to export the relevant EIs from Vision into a Python pickle file. You will have to specify
    a temporal upsampling factor (1 is allowed) for the EIs.
3. (Optional, skip if using default provided waveform prior means) If you want to specify your own waveform prior means,
    you can run `pca_init_basis_waveforms.py` to guess means from the data, or you can generate a Python pickle file with
    the appropriate format yourself. If you also want to use the L12 group-sparsity regularizer, you will also need to run
   `assign_basis_groups.py` to assign membership to the groups.
4. Run `fourier_nmf_single_cell_with_gaussian_prior.py` to run the decomposition algorithm.

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

### Creating your own waveform prior mean and sparsity groups

(Optional, but if you have EI lengths that have a different number of samples you will need to do this)

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

### Saving groups for group sparsity

If you are using the group sparsity L12 penalty, in addition to the basis prior means, you have to manually specify the groups. 
Since the PCA clustering method outputs basis vectors in random order, manual inspection and annotation is necessary.

Once you have you have decided the groups, use the following script to bind the groups to the basis prior means:

#### assign_basis_groups.py

The input csv each group on a different line, and the indices within each group separated by a comma. For example,

```shell script
0
1,2
```

corresponds to two groups. The first group contains basis waveform 0 alone, and the second group contains basis waveforms
1 and 2 together.

##### SYNOPSIS

```shell script
python assign_basis_groups.py <basis-pickle-path> <group-csv-path>
```

##### Example usage

This updates an already existing basis pickle at `input/basis_pickle.p` with the group assignments found in `input/group_assignment.csv`.

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
one amplitude + shifts step AND one waveform fitting step) to run. 3 is a reasonable value, more iterations will lead to
marginally better fits.
* --weight_reg, -w, float, default 1e-1. Weight to place on the group sparsity prior for the amplitudes + shifts optimization.
* -b, --before, positive integer value. Amount of pad/shifts forwards in time used in the shifts search component of the
amplitudes optimization. Determines the search space for the shifts search. In units of upsampled samples
* -a, --after, positive integer value. Amount of pad/shifts backwards in time used in the shifts search component of the
  amplitudes optimization. Determines the search space for the shifts search. In units of upsampled samples
* --grid_step, positive integer value, default value 5. Width of the steps in the coarse grid search over time during the amplitudes + shifts
optimization. Larger value runs faster, but may result in suboptimal time shifts
* --grid_top_n, positive integer value, default 4. Number of top candidates from the coarse time grid search to expand upon
for the fine search. Larger value runs slower, but may result in better time shifts.
* --fine_search_width, positive integer value, default 2. Width (on either side) of the fine time search. For each of the 
top n points from the coarse search, expand the neighborhood of shifts ranging from -fine_search_width to +fine_search_width
* --grid_batch_size, positive integer value, default 8192. Grid search batch size. Default value appears to be reasonable
and runs on GPUs with 12 Gb VRAM without issue. Increasing this number does not appear to drastically improve performance.
* --thresh, -t, float, default value 5.0. EI amplitude cutoff, below which the electrode is not used in the optimization
* --renormalize_loss, -r, store True, default False. Renormalizes the terms of the MSE loss during the waveform shape optimization
so that each electrode contributes approximately equally to the objective, even if the overall magnitudes of the recorded signals
are different. Leaving this False has the tendency to let a few large-amplitude channels dominate the waveform shape optimization,
and hence ignore the lower-amplitude channels entirely. Setting this to True has the tendency to increase the amount of noise
in the basis waveforms, as low-amplitude (noisier) channels are given a higher weight. All of the analyses in the paper set this
flag to True.
* --renormalize_penalty, -p, store True, default False. Renormalizes the weight on the (group) sparsity penalty so that the sparsity
is about the same for each channel (i.e. each channel should be about equally sparse; there should be no amplitude dependence). All
of the analyses in the paper set this flag to True.
* --group, -g, store True, default False. Use group sparsity regularization. If not set, then default to L1 regularization.
* --prior_weight, -pw, float, default value 1.0. Weight placed on the waveform shape Gaussian prior during the waveform shape
optimization. Smaller values will allow the waveforms to vary more from the prior means, while larger values tie the waveforms
more closely to the prior mean
* --prior_width, -psigma, float, default value 5.0. Width (in units of upsampled samples) of the Gaussian prior covariance. By
default we use an RBF kernel to construct the covariance matrix. The exact form of the covariance matrix can be changed in the source code.
* eps_cutoff, -e, float, default 1e-3. Cutoff for the FISTA amplitudes + shifts solver.
* --opt_iter, -o, positive integer default 10, maximum number of inner iterations fro the FISTA solver used to do the amplitudes + shifts
optimization. That optimization converges fairly fast, so 10 iterations is reasonable. More iterations will produce marginally better fits
but will be slower.


## Outputs

The output values are stored in a Python pickle file. The pickle file contains two distinct dictionaries. These dictionaries
must be loaded with consecutive calls of ```pickle.load()```.
1. The first dictionary is the optimization parameter dictionary. It contains all of the optimization hyperparameters for
record-keeping.
2. The second dictionary contains the results of the optimization. The keys in this dictionary are:
    * ```'mse'```: -> ```Dict``` the final MSE values over all waveforms at the end of the optimization
    * ```'waveforms``` -> ```np.ndarray``` the basis waveforms found by the optimization. Has shape ```(n_basis_waveforms, n_timepoints)```
    Note that because the waveforms are fit from random initialization, the compartments will occur in random order.
    * ```'decomposition'``` -> ```Dict[int, Tuple[np.ndarray, np.ndarray]]``` Dict mapping cell id integer to the decompositions.
    The first ```np.ndarray``` contains the amplitudes, and has shape ```(n_electrodes, n_basis_waveforms)```, with the order of
    the columns corresponding to the order of basis waveforms in ```waveforms```. The second ```np.ndarray``` contains the shifts,
    in units of supersampled samples, and has shape ```(n_electrodes, n_basis_waveforms)```. The order of the columns corresponds
    to the order of the basis waveforms in ```waveforms```. Negative is forward shift, and positive is a delay.
