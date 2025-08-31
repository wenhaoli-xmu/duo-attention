attn_pattern_name="lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10"
model_name="Meta-Llama-3.1-8B-Instruct"
model_path="/mnt/petrelfs/share_data/liwenhao/Llama-3.1-8B-Instruct"

TASKS=("scbench_kv" "scbench_prefix_suffix" "scbench_vt" "scbench_repoqa" "scbench_qa_eng" "scbench_qa_chn" "scbench_choice_eng"  "scbench_many_shot" "scbench_summary" "scbench_mf" "scbench_summary_with_needles" "scbench_repoqa_and_kv")

for task in ${TASKS[@]}; do
echo $task
torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    --master_addr localhost \
    --master_port 12345 \
    eval/scbench/run_scbench.py \
    --task $task \
    --model_name_or_path $model_path \
    --data_dir ./data \
    --output_dir ./results \
    --use_chat_template \
    --trust_remote_code \
    --max_seq_length 131_072 \
    --method duo-attn \
    --attn_load_dir "attn_patterns/${model_name}/${attn_pattern_name}" \
    --sparsity 0
done