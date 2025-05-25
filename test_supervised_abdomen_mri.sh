#!/bin/bash
# Fixed test script for supervised CoWPro model on abdominal MRI
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="supervised_cowpro_test"
DATASET='CHAOST2'  # Use base dataset name for validation
NWORKER=0

ALL_EV=(0 1 2 3 4) # 5-fold cross validation (0, 1, 2, 3, 4)

### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[2,3]' # setting 2: excluding kidneys in training set to test generalization capability

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[1,4]' 

###### Training configs (irrelevant in testing) ######
NSTEP=50000
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=10000 # interval for saving snapshot
SEED='1234'

###### Validation configs ######
SUPP_ID='[4]'  # using the additionally loaded scan as support

echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    PREFIX="test_supervised_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./exps_test_supervised/${CPT}_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir -p $LOGDIR
    fi

    # Updated path to the trained supervised model
    RELOAD_PATH_PATTERN="./exps_supervised/supervised_cowpro_${LABEL_SETS}/mySSL_supervised_train_CHAOST2_Supervised_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}_mindist4_CHAOST2_Supervised_sets_${LABEL_SETS}_1shot/*/snapshots/50000.pth"
    
    echo "Looking for model at: $RELOAD_PATH_PATTERN"
    
    # Check if model exists and get the actual path
    ACTUAL_RELOAD_PATH=""
    for path in $RELOAD_PATH_PATTERN; do
        if [ -f "$path" ]; then
            ACTUAL_RELOAD_PATH="$path"
            break
        fi
    done
    
    if [ -n "$ACTUAL_RELOAD_PATH" ] && [ -f "$ACTUAL_RELOAD_PATH" ]; then
        echo "Using model: $ACTUAL_RELOAD_PATH"
        
        # Use the fixed validation script
        python validation_supervised.py with \
        'modelname=dlfcn_res101' \
        'usealign=True' \
        'optim_type=sgd' \
        reload_model_path="$ACTUAL_RELOAD_PATH" \
        num_workers=$NWORKER \
        scan_per_load=-1 \
        label_sets=$LABEL_SETS \
        'use_wce=True' \
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
        support_idx=$SUPP_ID \
        val_wsize=2 \
        z_margin=0
        
        echo "Validation completed for fold $EVAL_FOLD"
        
    else
        echo "ERROR: Model not found for fold $EVAL_FOLD"
        echo "Checked path: $RELOAD_PATH_PATTERN"
        echo ""
        echo "Please verify that:"
        echo "1. Training completed successfully for fold $EVAL_FOLD"
        echo "2. The model was saved at iteration 50000"
        echo "3. The experiment directory structure matches the expected pattern"
        echo ""
        echo "You can check available models with:"
        echo "find ./exps_supervised -name '*.pth' -type f"
        
        # Continue to next fold instead of stopping
        continue
    fi
    
    echo "----------------------------------------"
done

echo ""
echo "Validation completed for all requested folds!"
echo "Results saved in: $LOGDIR"
echo ""
echo "Summary of what was processed:"
echo "- Dataset: $DATASET"
echo "- Label sets: $LABEL_SETS"
echo "- Excluded classes: $EXCLU"
echo "- Test folds: ${ALL_EV[@]}"
echo ""
echo "To view results:"
echo "1. Check Sacred logs in each experiment folder under $LOGDIR"
echo "2. Look for visualization images in interm_preds folders"
echo "3. Key metrics to look for in logs:"
echo "   - 'mar_val_batches_meanDice' for per-class Dice scores"
echo "   - 'mar_val_batches_classDice' for overall Dice scores"
echo "   - 'mar_val_batches_classPrec' and 'mar_val_batches_classRec' for precision/recall"
echo ""
echo "Example command to find your results:"
echo "find $LOGDIR -name 'run.json' -exec grep -l 'mar_val_batches_meanDice' {} \;"
