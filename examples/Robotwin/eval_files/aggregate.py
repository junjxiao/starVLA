import os
import sys
from datetime import datetime

def aggregate_results(root_path, output_file=None):
    """
    聚合指定路径下所有子文件夹的 _result.txt 结果，并保存到 TXT 文件
    
    Args:
        root_path (str): 根目录路径
        output_file (str): 输出文件路径，如果为 None 则自动生成
    """
    if not os.path.exists(root_path):
        print(f"❌ 路径不存在: {root_path}")
        return
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(root_path) 
               if os.path.isdir(os.path.join(root_path, d))]
    
    if not subdirs:
        print(f"⚠️  路径下没有子文件夹: {root_path}")
        return
    
    success_folders = []      # 包含 _result.txt 的文件夹
    missing_folders = []      # 不包含 _result.txt 的文件夹
    results = []              # 存储有效的结果值
    
    # 收集输出内容
    output_lines = []
    
    # 添加标题
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_lines.append(f"📊 聚合结果报告")
    output_lines.append(f"生成时间: {timestamp}")
    output_lines.append(f"根路径: {os.path.abspath(root_path)}")
    output_lines.append("=" * 60)
    
    for subdir in sorted(subdirs):
        folder_path = os.path.join(root_path, subdir)
        result_file = os.path.join(folder_path, "_result.txt")
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 检查是否有至少5行（索引4）
                if len(lines) >= 5:
                    fifth_line = lines[4].strip()  # 第五行（索引4）
                    if fifth_line:  # 确保不是空行
                        try:
                            value = float(fifth_line)
                            results.append(value)
                            success_folders.append(subdir)
                            line = f"✅ {subdir}: {value}"
                            output_lines.append(line)
                            print(line)
                        except ValueError:
                            line = f"⚠️  {subdir}: 第五行不是有效数字: '{fifth_line}'"
                            output_lines.append(line)
                            print(line)
                    else:
                        line = f"⚠️  {subdir}: 第五行为空"
                        output_lines.append(line)
                        print(line)
                else:
                    line = f"⚠️  {subdir}: 文件少于5行"
                    output_lines.append(line)
                    print(line)
                    
            except Exception as e:
                line = f"❌ {subdir}: 读取文件失败 - {e}"
                output_lines.append(line)
                print(line)
        else:
            missing_folders.append(subdir)
    
    # 统计信息
    output_lines.append("")
    output_lines.append("=" * 60)
    output_lines.append(f"📊 聚合结果统计:")
    output_lines.append(f"   总文件夹数: {len(subdirs)}")
    output_lines.append(f"   成功读取: {len(success_folders)}")
    output_lines.append(f"   缺失结果文件: {len(missing_folders)}")
    
    if results:
        avg_result = sum(results) / len(results)
        output_lines.append(f"   平均结果: {avg_result:.6f}")
        print(f"\n✅ 平均结果: {avg_result:.6f}")
    else:
        output_lines.append("   平均结果: N/A (无有效数据)")
        print(f"\n⚠️  平均结果: N/A (无有效数据)")
    
    # 成功文件夹列表
    output_lines.append("")
    output_lines.append("📁 包含 _result.txt 的文件夹:")
    if success_folders:
        for folder in sorted(success_folders):
            output_lines.append(f"   • {folder}")
    else:
        output_lines.append("   (无)")
    
    # 缺失文件夹列表
    output_lines.append("")
    output_lines.append("📁 不包含 _result.txt 的文件夹:")
    if missing_folders:
        for folder in sorted(missing_folders):
            output_lines.append(f"   • {folder}")
    else:
        output_lines.append("   (无)")
    
    # 确定输出文件路径
    if output_file is None:
        output_file = os.path.join(root_path, "aggregated_results.txt")
    
    # 保存到文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\n💾 结果已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
    
    return output_file

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("用法: python aggregate_results.py <结果根路径> [输出文件路径]")
    #     print("示例1: python aggregate_results.py /path/to/results")
    #     print("示例2: python aggregate_results.py /path/to/results /path/to/output.txt")
    #     sys.exit(1)
    
    # root_path = sys.argv[1]
    # output_file = sys.argv[2] if len(sys.argv) > 2 else None
    root_path = '/mnt/workspace/junjin/code/starVLAPretrain/outputs/robotwin/0203_robotwin2_mix_QwenJAT_vggt_sft_64GPUs_OAR_balance_1w_test_num_100_step50000/demo_clean'
    aggregate_results(root_path, None)
