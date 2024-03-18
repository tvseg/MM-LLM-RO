GPUS=1
GPU_DEVICE=1
WORKDIR='/home/gpuadmin/yujin/ro-llama/work_dir/BC_NC/'
CHECKPOINTLIST='v9.0_Unimodal v9.1_Multimodal_llama2' 
PLIST='1.00'

for P in $PLIST
do
    for CHECK in $CHECKPOINTLIST
    do
        CHECK_P=$CHECK'_'$P
        echo -E $CHECK_P
        if [[ $CHECK_P == *"llama"* ]] ; then 
            CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --logdir $WORKDIR$CHECK_P --context True --p_data $P --n_prompts 2 --context_length 8 
            CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --pretrained_dir $WORKDIR$CHECK_P --p_data $P --context True --n_prompts 2 --context_length 8 --test_mode 1 
            CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --pretrained_dir $WORKDIR$CHECK_P --p_data $P --context True --n_prompts 2 --context_length 8 --test_mode 2 
            CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --pretrained_dir $WORKDIR$CHECK_P --p_data $P --context True --n_prompts 2 --context_length 8 --test_mode 3 
        else
            CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --logdir $WORKDIR$CHECK_P --p_data $P 
            CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --pretrained_dir $WORKDIR$CHECK_P --p_data $P --test_mode 1 
            CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --pretrained_dir $WORKDIR$CHECK_P --p_data $P --test_mode 2 
            CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --pretrained_dir $WORKDIR$CHECK_P --p_data $P --test_mode 3 
        fi
    done
done

