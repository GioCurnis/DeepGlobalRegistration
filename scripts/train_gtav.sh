#! /bin/bash
export PATH_POSTFIX=$1
export MISC_ARGS=$2

export DATA_ROOT="/home/curnis/result/fcgf/gtav"
export DATASET=${DATASET:-GTAVNMPairDataset}
export KITTI_PATH="/media/RAIDONE/curnis/GTAVDataset"
export MODEL=${MODEL:-ResUNetBN2C}
export MODEL_N_OUT=${MODEL_N_OUT:-32}
export FCGF_WEIGHTS="/tmp/pycharm_project_153/pretrained/KITTI-v0.3-ResUNetBN2C-conv1-5-nout32.pth"
export INLIER_MODEL=${INLIER_MODEL:-ResUNetBN2C}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-2}
export MAX_EPOCH=${MAX_EPOCH:-1}
export BATCH_SIZE=${BATCH_SIZE:-8}
export ITER_SIZE=${ITER_SIZE:-1}
export VOXEL_SIZE=${VOXEL_SIZE:-0.3}
export POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER=${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER:-4}
export CONV1_KERNEL_SIZE=${CONV1_KERNEL_SIZE:-5}
export EXP_GAMMA=${EXP_GAMMA:-0.99}
export RANDOM_SCALE=${RANDOM_SCALE:-True}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export OUT_DIR=${DATA_ROOT}/${DATASET}-v${VOXEL_SIZE}/${INLIER_MODEL}/${OPTIMIZER}-lr${LR}-e${MAX_EPOCH}-b${BATCH_SIZE}i${ITER_SIZE}-modelnout${MODEL_N_OUT}${PATH_POSTFIX}/${TIME}

export PYTHONUNBUFFERED="True"

echo $OUT_DIR

mkdir -m 755 -p $OUT_DIR

LOG=${OUT_DIR}/log_${TIME}.txt

echo "Host: " $(hostname) | tee -a $LOG
echo "Conda " $(which conda) | tee -a $LOG
echo $(pwd) | tee -a $LOG
echo "" | tee -a $LOG
git diff | tee -a $LOG
echo "" | tee -a $LOG
nvidia-smi | tee -a $LOG

# Training
python train.py \
	--weights ${FCGF_WEIGHTS} \
	--dataset ${DATASET} \
	--feat_model ${MODEL} \
	--feat_model_n_out ${MODEL_N_OUT} \
	--feat_conv1_kernel_size ${CONV1_KERNEL_SIZE} \
	--inlier_model ${INLIER_MODEL} \
	--optimizer ${OPTIMIZER} \
	--lr ${LR} \
	--batch_size ${BATCH_SIZE} \
	--val_batch_size ${BATCH_SIZE} \
	--iter_size ${ITER_SIZE} \
	--max_epoch ${MAX_EPOCH} \
	--voxel_size ${VOXEL_SIZE} \
	--out_dir ${OUT_DIR} \
	--use_random_scale ${RANDOM_SCALE} \
	--kitti_dir ${KITTI_PATH} \
	--success_rte_thresh 2 \
	--success_rre_thresh 5 \
	--positive_pair_search_voxel_size_multiplier ${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER} \
	$MISC_ARGS 2>&1 | tee -a $LOG

