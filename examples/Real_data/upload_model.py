import os
os.environ['MODELSCOPE_API_TOKEN']='ms-5902e780-0218-4815-90d3-ed05bc1fbdc3'
from modelscope.hub.snapshot_download import snapshot_download
import modelscope
from modelscope.hub.api import HubApi

import os
import shutil
from pathlib import Path

# # 配置
# SRC_DIR = Path("/mnt/workspace/junjin/code/starVLA/checkpoints/0129_real_put_toy_in_cabinet_Qwen3vlGR00T_vggt_use_state_cross_bs8")                     # 原始模型目录
# TARGET_MODEL_FILE = "steps_10000_pytorch_model.pt" # 你要保留的模型文件名
# repo_id = "junjxiao/starvla_put_toy_in_cabinet_state_10000"

# UPLOAD_DIR = Path("/mnt/workspace/junjin/code/starVLA/checkpoints/tmp")       # 干净上传目录
# # 清理旧目录
# if UPLOAD_DIR.exists():
#     shutil.rmtree(UPLOAD_DIR)
# UPLOAD_DIR.mkdir()

# # 复制 checkpoints 目录（只保留指定模型）
# (UPLOAD_DIR / "checkpoints").mkdir()
# shutil.copy2(SRC_DIR / "checkpoints" / TARGET_MODEL_FILE, UPLOAD_DIR / "checkpoints" / TARGET_MODEL_FILE)

# # 复制其他所有文件（排除 wandb 和 logs）
# for item in SRC_DIR.iterdir():
#     if item.name in ["wandb", "logs", "checkpoints"]:
#         continue
#     if item.is_file():
#         shutil.copy2(item, UPLOAD_DIR / item.name)
#     elif item.is_dir():
#         shutil.copytree(item, UPLOAD_DIR / item.name)

# print(f"Clean upload directory prepared at: {UPLOAD_DIR}")


# # SRC_DIR = Path("/mnt/workspace/junjin/code/starVLA/checkpoints/0128_real_clean_the_table_Qwen3vlGR00T_vggt_cross_bs8") 
# api = HubApi()
# api.upload_folder(
# repo_id=repo_id,
# folder_path=UPLOAD_DIR,
# commit_message='Initial release',
# repo_type='model'
# )

api = HubApi()
api.upload_folder(
repo_id='junjxiao/robotwin',
folder_path='/mnt/workspace/junjin/code/starVLA/checkpoints/robotwin',
commit_message='Initial release',
repo_type='model'
)