import os
import sys
from pathlib import Path

# 将项目根目录添加到 python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from app.pipelines.geo_pipeline import GeoInversionPipeline
from app.core.config import INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR

def main():
    # 1. 确保基础目录存在
    for d in [INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # 定义更细致的输出子目录
    pdf_extraction_dir = os.path.join(PROCESSED_DIR, "4_pdf_extraction")
    report_output_dir = os.path.join(PROCESSED_DIR, "5_final_report")
    
    # 定义新的 PDF 输入目录 (从 Part 1 下载结果读取)
    pdf_input_dir = os.path.join(PROCESSED_DIR, "3_paper_download", "pdfs")
    
    for d in [pdf_extraction_dir, report_output_dir]:
        os.makedirs(d, exist_ok=True)
        
    # 检查 PDF 输入目录
    if not os.path.exists(pdf_input_dir):
        print(f"Warning: PDF input directory not found: {pdf_input_dir}")
        print("Creating it for safety, but it might be empty if Part 1 hasn't run.")
        os.makedirs(pdf_input_dir, exist_ok=True)
    
    # 2. 用户输入配置 (假设用户把 example 里的东西放到了 input 里)
    user_analysis_json = os.path.join(INPUT_DIR, "user_inversion_analysis.json")
    # 如果 input 里没有，尝试找一下 examples/test_geo_info.json 作为 fallback 或者 dummy
    if not os.path.exists(user_analysis_json):
         user_analysis_json = os.path.join(PROJECT_ROOT, "examples", "test_geo_info.json")
         
    final_report_path = os.path.join(report_output_dir, "LangChain_Comprehensive_Report.md")
    
    # 3. 检查输入
    if not os.path.exists(user_analysis_json):
        print(f"Warning: User analysis file not found: {user_analysis_json}")
        print("Creating dummy user analysis for demonstration...")
        user_analysis_json = os.path.join(INPUT_DIR, "user_inversion_analysis.json")
        import json
        with open(user_analysis_json, 'w', encoding='utf-8') as f:
            json.dump({
                "user_analysis": "这里是用户提供的反演结果描述：在深度100km处发现明显低速异常..."
            }, f, ensure_ascii=False, indent=2)
        print(f"Created dummy analysis at: {user_analysis_json}")

    # 4. 初始化全流程管线
    pipeline = GeoInversionPipeline(
        pdf_input_dir=pdf_input_dir, # 修改为从 Part 1 的下载目录读取
        extraction_output_dir=pdf_extraction_dir,
        final_report_path=final_report_path
    )
    
    # 5. 运行
    print("\n=== 开始执行 GeoInversionPipeline ===")
    try:
        pipeline.run(user_analysis_json=user_analysis_json)
    except Exception as e:
        print(f"\nFAILURE: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
