# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Patch

# # ================= 1. 数据准备 (移除 Nora 和 WorldVLA) =================
# methods = [
#     'OpenVLA', 'OpenVLA-OFT', r'$\pi$0', r'$\pi$0-fast',
#     'UniVLA', 'Ours'
# ]

# # LIBERO-plus spatial (深色部分，作为基底)
# libero_plus = np.array([19.4, 84.0, 60.7, 74.4, 55.5, 90.8])

# # LIBERO spatial (总高度)
# libero_total = np.array([84.7, 97.6, 98.0, 96.4, 96.5, 98.8])

# # 计算浅色部分的高度 (LIBERO - LIBERO-plus)
# libero_diff = libero_total - libero_plus

# # ================= 2. 配色方案 =================
# base_colors = [
#     '#5DADE2',  # OpenVLA - 蓝
#     '#58D68D',  # OpenVLA-OFT - 绿
#     '#F5B041',  # pi0 - 橙
#     '#AF7AC5',  # pi0-fast - 紫
#     '#EC7063',  # UniVLA - 浅红
#     '#E74C3C'   # Ours - 深红/珊瑚红 (Highlight)
# ]

# light_colors = [
#     '#AED6F1',  # 浅蓝
#     '#ABEBC6',  # 浅绿
#     '#FDEBD0',  # 浅橙
#     '#D7BDE2',  # 浅紫
#     '#F5B7B1',  # 浅红
#     '#FADBD8'   # 浅珊瑚红
# ]

# # ================= 3. 绘图设置 =================
# # figsize=(9, 9) -> 1:1 正方形
# # 如果需要高大于宽 (1:1.8)，请改为 figsize=(5, 9)
# fig, ax = plt.subplots(figsize=(9, 9)) 

# x = np.arange(len(methods))
# width = 0.6  

# # ================= 4. 绘制堆叠柱状图 =================
# bars_bottom = ax.bar(x, libero_plus, width, 
#                      color=base_colors, 
#                      edgecolor='white', linewidth=1.0)

# bars_top = ax.bar(x, libero_diff, width, 
#                   bottom=libero_plus, 
#                   color=light_colors, 
#                   edgecolor='white', linewidth=1.0)

# # ================= 5. 美化与布局 =================
# ax.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold')
# ax.set_ylim(0, 110) 
# ax.set_yticks(np.arange(0, 111, 20))
# ax.tick_params(axis='y', labelsize=14)

# # --- X 轴设置 ---
# ax.set_xticks(x)
# ax.set_xticklabels([]) # 依然隐藏具体方法名，因为右侧有图例
# ax.set_xlabel('Spatial', fontsize=16, fontweight='bold') # 添加 "Spatial" 标签
# ax.tick_params(axis='x', length=5, width=1.5) # 显示 X 轴刻度线

# # 网格线
# ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC', zorder=0)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(True) # 恢复底部边框
# ax.spines['bottom'].set_linewidth(1.5) # 加粗底部边框

# # ================= 6. 图例设置在最右侧 =================
# custom_handles = []
# for i, method in enumerate(methods):
#     p = Patch(facecolor=base_colors[i], edgecolor='black', linewidth=0.5, label=method)
#     custom_handles.append(p)

# ax.legend(handles=custom_handles, 
#           loc='center left', 
#           bbox_to_anchor=(1.02, 0.5), 
#           frameon=False,   
#           fontsize=14,     
#           ncol=1)          

# # ================= 7. 调整布局 =================
# # rect=[left, bottom, right, top]
# # bottom=0.0: 不需要额外留白给底部文字了
# # right=0.85: 给右侧图例留出空间
# plt.tight_layout(rect=[0, 0, 0.85, 1]) 

# # 保存图像
# plt.savefig('libero_stacked_spatial_label.png', dpi=300, bbox_inches='tight')
# plt.savefig('libero_stacked_spatial_label.pdf', bbox_inches='tight')

# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ================= 1. 数据准备 =================
methods = [
    'OpenVLA', 'OpenVLA-OFT', r'$\pi$0', r'$\pi$0-fast',
    'UniVLA', 'Ours'
]

libero_plus = np.array([19.4, 84.0, 60.7, 74.4, 55.5, 90.8])
libero_total = np.array([84.7, 97.6, 98.0, 96.4, 96.5, 98.8])
libero_diff = libero_total - libero_plus

# ================= 2. 配色方案 =================
base_colors = ['#5DADE2', '#58D68D', '#F5B041', '#AF7AC5', '#EC7063', '#E74C3C']
light_colors = ['#AED6F1', '#ABEBC6', '#FDEBD0', '#D7BDE2', '#F5B7B1', '#FADBD8']

# ================= 3. 绘图设置 =================
fig, ax = plt.subplots(figsize=(9, 9)) 
x = np.arange(len(methods))
width = 0.6  

# ================= 4. 绘制堆叠柱状图 =================
bars_bottom = ax.bar(x, libero_plus, width, color=base_colors, edgecolor='white', linewidth=1.0)
bars_top = ax.bar(x, libero_diff, width, bottom=libero_plus, color=light_colors, edgecolor='white', linewidth=1.0)

# ================= 5. 绘制差值箭头 (核心修改) =================
for i in range(len(methods)):
    # 计算箭头的起始Y (Libero-Plus顶部) 和 结束Y (Libero Total顶部)
    y_start = libero_plus[i]
    y_end = libero_total[i]
    
    # 箭头的X位置：放在柱子的右边缘稍微偏右一点，或者柱子中心
    # 这里我们选择放在柱子中心的右侧一点点，避免压住柱子主体
    x_pos = x[i] + width / 2 + 0.05 
    
    # 只有当差值足够大时才绘制箭头，避免视觉混乱
    if libero_diff[i] > 5:
        # 绘制双向箭头
        ax.annotate(
            '', 
            xy=(x_pos, y_start), 
            xytext=(x_pos, y_end),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
        )
        
        # 在箭头中间添加文字 "Gap" 或数值
        mid_y = (y_start + y_end) / 2
        + width / 2 + 0.05 
        ax.text(
            x[i]- width / 2 - 0.05  , # 文字稍微再往右一点
            mid_y,
            f'{libero_diff[i]:.1f}', # 显示差值
            ha='left', va='center',
            fontsize=20,
            color='black',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none') # 白色半透明背景，防止看不清
        )

# ================= 6. 美化与布局 =================
ax.set_ylabel('Success Rate (%)', fontsize=26, fontweight='bold')
ax.set_ylim(0, 100) 
ax.set_yticks(np.arange(0, 101, 20))
ax.tick_params(axis='y', labelsize=24)

# X轴设置
ax.set_xticks(x)
ax.set_xticklabels([]) 
ax.set_xlabel('Spatial', fontsize=24, fontweight='bold')
ax.tick_params(axis='x', length=5, width=1.5)

# 网格线与边框
ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(1.5)

# ================= 7. 图例设置 =================
# custom_handles = []
# for i, method in enumerate(methods):
#     p = Patch(facecolor=base_colors[i], edgecolor='black', linewidth=0.5, label=method)
#     custom_handles.append(p)

# ax.legend(handles=custom_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=14, ncol=1)

# ================= 8. 调整布局 =================
plt.tight_layout(rect=[0, 0, 0.85, 1]) 

plt.savefig('libero_stacked_with_gap_arrows.png', dpi=300, bbox_inches='tight')
plt.savefig('libero_stacked_with_gap_arrows.pdf', bbox_inches='tight')
plt.show()
