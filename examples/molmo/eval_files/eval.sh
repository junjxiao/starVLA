# source /mnt/workspace/zengshuang.zs/molmospaces/.venv/bin/activate
# export MUJOCO_GL=osmesa
export MLSPACES_ASSETS_DIR=/mnt/workspace/zengshuang.zs/molmospaces_assets
export PYTHONPATH=$(pwd):${PYTHONPATH}
benchmark_dir="/mnt/workspace/zengshuang.zs/molmospaces_assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
checkpoint_path="/mnt/workspace/zengshuang.zs/output/roboarena/0315_roboarena_QwenJAT_sft_joint_worker2/checkpoints/steps_100000_pytorch_model.pt"
output_dir="./outputs/molmospace/0315_roboarena_QwenJAT_sft_joint_worker2_steps_100000"

task_horizon_steps=450
host="localhost"
port=9980
xvfb-run -a python examples/molmo/eval_files/eval.py \
    --args.benchmark-dir ${benchmark_dir} \
    --args.host "$host" \
    --args.port $port \
    --args.output-dir "$output_dir" \
    --args.task-horizon-steps "$task_horizon_steps" \
    --args.checkpoint-path "$checkpoint_path" \