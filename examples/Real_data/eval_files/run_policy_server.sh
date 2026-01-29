
your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0129_real_put_toy_in_cabinet_Qwen3vlGR00T_vggt_use_state_cross_bs8/checkpoints/steps_1000_pytorch_model.pt
output_dir=/mnt/workspace/junjin/code/starVLA/outputs/real/put_toy_in_cabinet_test
CUDA_VISIBLE_DEVICES=1 python ./examples/Real_data/eval_files/policy_server.py \
    --ckpt_path ${your_ckpt} \
    --use_bf16 \
    --output_dir ${output_dir}