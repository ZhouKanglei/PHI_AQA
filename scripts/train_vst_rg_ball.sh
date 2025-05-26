# !/bin/bash
date
# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# go to the directory where the script is located
cd ~/zkl/Codes/PHI_AQA
pwd

# setting
path=~/zkl/Data/AQA/GDLT_data/VST/RG
gpu=2


# first training phase: initialization. Note that this initialization is slightly different from the original paper, and you may need to adjust the parameters based on your specific requirements. My parameters may not be the best and there may be better parameters for initialization.
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --video-path ${path}/swintx_avg_fps25_clip32 \
    --train-label-path ${path}/train.txt \
    --test-label-path ${path}/test.txt  \
    --model-name phi \
    --action-type Ribbon \
    --lr 1e-2 --epoch 100 \
    --n_encoder 1 --n_decoder 2 --n_query 4 --alpha 1 --margin 1 --lr-decay cos --decay-rate 1e-2 --dropout 0.3 \
    --loss_align 1 --activate-type 2 --n_head 1 --hidden_dim 256 --flow_hidden_dim 256 \
    --exp-name first_phase

# second training phase: load the best checkpoint from the first training phase
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --video-path ${path}/swintx_avg_fps25_clip32 \
    --train-label-path ${path}/train.txt \
    --test-label-path ${path}/test.txt  \
    --model-name phi \
    --action-type Ball \
    --lr 1e-2 --epoch 200 \
    --n_encoder 1 --n_decoder 2 --n_query 4 --alpha 1 --margin 1 --lr-decay cos --decay-rate 1e-2 --dropout 0.3 \
    --loss_align 1 --activate-type 2 --n_head 1 --hidden_dim 256 --beta 0.01 --flow_hidden_dim 256 \
    --ckpt outputs/phi/Ribbon/first_phase/best.pkl \
    --exp-name second_phase

