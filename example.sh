#!/bin/bash

L1_REG=1e-1
THRESH=5.0
BEFORE=40
AFTER=80
##############################################################

##### These should stay constant across pieces, since #######
##### they specify the waveform prior #######################
PRIOR_WEIGHT=5e-3
PRIOR_WIDTH=10
NO_SPAT_ITER=3
REG_FLAGS="-r -p -g" # use group sparsity penalty, renormalize data waveforms, and normalize loss
############################################################

##############################################################

EVERYTHING_IDPATH=/path/to/your/csv

#### EXTRACT THE DATA OUT OF VISION #################################################
echo python ${CODEDIR}/create_data_pickle.py $DSPATH $DSNAME $EVERYTHING_IDPATH $DATA_PICKLE
python ${CODEDIR}/create_data_pickle.py $DSPATH $DSNAME $EVERYTHING_IDPATH $DATA_PICKLE

##### RUN DECOMPOSITION ##############################################################
OUTPUT_FNAME=$TEMPDIR"/L12_${L1_REG}_thresh_${THRESH}.p"
echo python ${CODEDIR}/fourier_nmf_single_cell_with_gaussian_prior.py $DATA_PICKLE $BASIS_PICKLE $OUTPUT_FNAME  -m $NO_SPAT_ITER -w $L1_REG --grid_step 5 --grid_top_n 5 --fine_search_width 2 --grid_batch_size 4096 --thresh $THRESH $REG_FLAGS --prior_weight $PRIOR_WEIGHT --prior_width $PRIOR_WIDTH --opt_iter 10 -b $BEFORE -a $AFTER
python ${CODEDIR}/fourier_nmf_single_cell_with_gaussian_prior.py $DATA_PICKLE $BASIS_PICKLE $OUTPUT_FNAME  -m $NO_SPAT_ITER -w $L1_REG --grid_step 5 --grid_top_n 5 --fine_search_width 2 --grid_batch_size 4096 --thresh $THRESH $REG_FLAGS --prior_weight $PRIOR_WEIGHT --prior_width $PRIOR_WIDTH --opt_iter 10 -b $BEFORE -a $AFTER