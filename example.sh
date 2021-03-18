#!/bin/bash

PROG_DIR="/Volumes/Lab/Users/ericwu/development/ei_decomposition"

CELL_ID_PICK_PROG="${PROG_DIR}/generate_typed_cell_id_list.py"
DATA_PREP_PROG="${PROG_DIR}/create_data_pickle.py"

BASIS_MANUAL_PROG="${PROG_DIR}/save_initialized_waveforms_basis.py"
BASIS_CLUSTER_PROG="${PROG_DIR}/pca_init_basis_waveforms.py"
BASIS_GROUP_PROG="${PROG_DIR}/assign_basis_groups.py"

TWO_STAGE_PROG="${PROG_DIR}/two_stage_fourier_decomp.py"

DS_PATH="/Volumes/Lab/Users/ericwu/yass-ei/2018-03-01-0/data001"
DS_NAME="data001"

OUTPUT_PATH="/Volumes/Scratch/Users/wueric/useless/2018-03-01-0/data001/off_parasol"

mkdir -p ${OUTPUT_PATH}

CELLTYPE_CSV_PATH="${OUTPUT_PATH}/off_parasol_list.csv"
EI_DATA_PATH="${OUTPUT_PATH}/off_parasol_raw_ei.p"
BASIS_OUTPUT_FNAME="${OUTPUT_PATH}/off_parasol_pca_basis.p"
OUTPUT_FNAME="${OUTPUT_PATH}/off_parasol_spat_pca_decomp.p"
GROUP_FILE="${OUTPUT_PATH}/off_parasol_groups.csv"

L1_REG=1e-1
SOB_REG=1e-2
THRESH=5
REG_FLAGS="-r -p -g"

UPSAMPLE=5
BEFORE=100
AFTER=200

EPS_CUTOFF=5e-2
CONVERGENCE_FLAGS=""

#########################################
# Run the three commands below first ####
# Then exit #############################

# first prepare the cell list
echo python ${CELL_ID_PICK_PROG} $DS_PATH $DS_NAME $CELLTYPE_CSV_PATH "OFF parasol"
python ${CELL_ID_PICK_PROG} $DS_PATH $DS_NAME $CELLTYPE_CSV_PATH "OFF parasol"

# then extract the EI data from Vision
echo python $DATA_PREP_PROG $DS_PATH $DS_NAME $CELLTYPE_CSV_PATH $EI_DATA_PATH
python $DATA_PREP_PROG $DS_PATH $DS_NAME $CELLTYPE_CSV_PATH $EI_DATA_PATH

# initialize basis vectors with PCA clustering
echo python ${BASIS_CLUSTER_PROG} $EI_DATA_PATH $BASIS_OUTPUT_FNAME -u $UPSAMPLE -b $BEFORE -a $AFTER -t $THRESH -l 400 -p 5
python ${BASIS_CLUSTER_PROG} $EI_DATA_PATH $BASIS_OUTPUT_FNAME -u $UPSAMPLE -b $BEFORE -a $AFTER -t $THRESH -l 400 -p 5

exit;

#############################################################################
# Now manually annotate the group assignment CSV located at ${GROUP_FILE} ###
# When you're done, comment out the first three commands and the exit #######
# and then run the below commands ###########################################

# after manually annotating the groups
echo python ${BASIS_GROUP_PROG} $BASIS_OUTPUT_FNAME ${GROUP_FILE}
python ${BASIS_GROUP_PROG} $BASIS_OUTPUT_FNAME ${GROUP_FILE}

NO_SPAT_ITER=3
echo python ${TWO_STAGE_PROG} ${EI_DATA_PATH} ${OUTPUT_FNAME} --initialize_basis ${BASIS_OUTPUT_FNAME} -w ${L1_REG} -m ${NO_SPAT_ITER} --grid_step 8 --fine_search_width 4 ${REG_FLAGS} -e $EPS_CUTOFF $CONVERGENCE_FLAGS
python ${TWO_STAGE_PROG} ${EI_DATA_PATH} ${OUTPUT_FNAME} --initialize_basis ${BASIS_OUTPUT_FNAME} -w ${L1_REG} -m ${NO_SPAT_ITER} --grid_step 8 --fine_search_width 4 ${REG_FLAGS} -e $EPS_CUTOFF $CONVERGENCE_FLAGS

