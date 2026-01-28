
your_ckpt=/mnt/workspace/zengshuang.zs/output/libero_all/1227_libero4in1_QwenGR00T_sft/checkpoints/steps_30000_pytorch_model.pt
base_port=9883
export star_vla_python=/mnt/workspace/junjin/conda/starVLA/bin/python

export DEBUG=1

CUDA_VISIBLE_DEVICES=3 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16