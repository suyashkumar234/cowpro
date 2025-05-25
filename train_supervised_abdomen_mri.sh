#!/bin/bash
# Training script for supervised CoWPro model
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="supervised_cowpro"
DATASET='CHAOST2_Supervised'  # or 'SABS_Supervised'
NWORKER=0

ALL_EV=(0 1 2 3 4) # 5-fold cross validation
MIN_SLICE_DIST=4    # Minimum distance between support and query slices
MAX_DIST_RATIO=0.1667  # 1/6 as decimal

### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[2,3]' # setting 2: excluding kidneys in training set to test generalization

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[1,4]' 

###### Training configs ######
NSTEP=50000  # Reduced from self-supervised training
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=10000 # interval for saving snapshot
SEED='1234'

echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    PREFIX="supervised_train_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}_mindist${MIN_SLICE_DIST}"
    echo $PREFIX
    LOGDIR="./exps_supervised/${CPT}_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir -p $LOGDIR
    fi

    python training_supervised.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    'use_tversky=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    min_fg_data=1 \
    seed=$SEED \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    min_slice_distance=$MIN_SLICE_DIST \
    max_distance_ratio=$MAX_DIST_RATIO \
    which_aug='aug_v3' \
    print_interval=200
done
