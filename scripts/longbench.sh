model=$1
task=$2
attn_pattern=$3
sparsity=$4


srun -p Intern5 --quotatype spot -N1 -n1 --gpus-per-task 1 \
python -u eval/LongBench/pred.py \
    --model $model --task $task \
    --method duo_attn \
    --attn_load_dir ${attn_pattern} \
    --sparsity $sparsity \
    --sink_size 64 \
    --recent_size 256
