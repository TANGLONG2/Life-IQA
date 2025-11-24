# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49935  main.py \
# --cfg [CONFIG_PATH] \
# --data-path [YOUR_DATA_PATH] \
# --output [LOG_PATH] \
# --tag [REMARK_TAG] \
# --repeat \
# --rnum [TARGET_REPEAT_NUM]
# --num_experts NUM \
# --top_k NUM\


# examples:


export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49941  main.py \
--cfg 'configs/swin/swin_tid2013.yaml' \
--data-path 'Datasets/TID2013/tid2013' \
--output 'Output' \
--tag 'TID2013_test' \
--repeat \
--rnum 10 \
--num_experts 8 \
--top_k 2\









