
your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0128_real_clean_the_table_Qwen3vlGR00T_vggt_cross_bs8/checkpoints/steps_1000_pytorch_model.pt
output_dir=/mnt/workspace/junjin/code/starVLA/outputs/real/clean_the_table_test
CUDA_VISIBLE_DEVICES=1 python ./examples/Real_data/eval_files/policy_server.py \
    --ckpt_path ${your_ckpt} \
    --use_bf16 \
    --output_dir ${output_dir}