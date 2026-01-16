import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# 添加当前目录到 sys.path，确保能导入 main.py 中的类
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from main1 import EarthPhysicsAgent
except ImportError:
    # 尝试从当前包导入
    from agents.main1 import EarthPhysicsAgent

def run_workflow(image_path: str, geo_info_path: str, output_dir: Path):
    """
    运行 Workflow I: 图像分析与初步解释
    
    Args:
        image_path: 图像文件路径
        geo_info_path: 地理信息 JSON 文件路径
        output_dir: 输出目录 Path 对象
    
    Returns:
        dict: 包含 geo_info, image_analysis, wf1_report 的字典
    """
    print(f"\n>>> [Workflow I] Starting Image Analysis (Full EarthPhysicsAgent)...")
    print(f"    图像: {image_path}")
    print(f"    地理信息: {geo_info_path}")

    # 加载地理信息
    if not Path(geo_info_path).exists():
         raise FileNotFoundError(f"[Error] 地理信息文件不存在: {geo_info_path}")
         
    with open(geo_info_path, 'r', encoding='utf-8') as f:
        geo_info = json.load(f)
    
    # 验证必需字段
    required_fields = ['longitude_range', 'latitude_range', 'depth_range', 'region_name']
    for field in required_fields:
        if field not in geo_info:
            raise ValueError(f"地理信息缺少必需字段: {field}")
    
    # 创建Agent并处理
    # Workflow I 不需要文献检索功能
    agent = EarthPhysicsAgent(enable_literature=False)
    print(f">>> [Workflow I] Running EarthPhysicsAgent logic...")
    result = agent.run_workflow_i(image_path, geo_info, branches=3)
    
    image_analysis = result.get("image_analysis", {})
    wf1_report = result.get("final_report", "")
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "image_analysis.json", "w", encoding="utf-8") as f:
        json.dump(image_analysis, f, ensure_ascii=False, indent=2)
        
    with open(output_dir / "wf1_report.md", "w", encoding="utf-8") as f:
        f.write(wf1_report)
        
    # Save other artifacts from full workflow
    if "hypotheses" in result:
        with open(output_dir / "hypotheses.json", "w", encoding="utf-8") as f:
            json.dump(result["hypotheses"], f, ensure_ascii=False, indent=2)
    
    if "evaluation" in result:
        with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(result["evaluation"], f, ensure_ascii=False, indent=2)
            
    print(f">>> [Workflow I] Completed. Report saved to {output_dir / 'wf1_report.md'}")
    
    return {
        "geo_info": geo_info,
        "image_analysis": image_analysis,
        "wf1_report": wf1_report
    }

def main():
    """Workflow I 独立执行入口"""
    base_dir = Path(__file__).resolve().parent.parent # agents -> langchain_version
    
    # 优先从 data/input 读取
    input_dir = base_dir / "data" / "input"
    default_image = str((input_dir / "image.png").resolve())
    default_geo_info = str((input_dir / "test_geo_info.json").resolve())
    
    # 如果 input 里没有，回退到 examples (兼容旧逻辑)
    if not Path(default_image).exists():
        default_image = str((base_dir / "examples" / "image.png").resolve())
    if not Path(default_geo_info).exists():
        default_geo_info = str((base_dir / "examples" / "test_geo_info.json").resolve())
        
    default_output_dir = str((base_dir / "data" / "processed").resolve())

    parser = argparse.ArgumentParser(description="地球物理反演解释Agent - Workflow I (初步解释)")
    parser.add_argument("--image", default=default_image, help="反演结果图像路径")
    parser.add_argument("--geo-info", default=default_geo_info, help="地理信息JSON文件路径")
    parser.add_argument("--output-dir", default=default_output_dir, help="输出目录路径")
    parser.add_argument("--branches", type=int, default=3, help="思维树分支数")
    
    args = parser.parse_args()
    
    try:
        run_workflow(args.image, args.geo_info, Path(args.output_dir))
    except Exception as e:
        print(f"\n[Error] 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
