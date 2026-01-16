import sys
import os
import json
import asyncio
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure langchain_version is in path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import Workflow 1 components
from agents.workflow1 import run_workflow as run_image_analyze

# Import Workflow 2 components (Now from download.py)
from agents.download import run_search_and_score as run_literature_search, PaperInfo

# Import Workflow 3 components (Now from run_sep_download.py)
from agents.run_sep_download import run_download_workflow as run_paper_download

# ==========================================
# Main Workflows
# ==========================================

async def main():
    parser = argparse.ArgumentParser(description="Integrated Workflow (I + II + III)")
    
    base_dir = Path(__file__).resolve().parent
    
    # 优先从 data/input 读取
    input_dir = base_dir / "data" / "input"
    default_image = str((input_dir / "image.png").resolve())
    default_geo_info = str((input_dir / "test_geo_info.json").resolve())
    
    # 如果 input 里没有，回退到 examples
    if not Path(default_image).exists():
        default_image = str((base_dir / "examples" / "image.png").resolve())
    if not Path(default_geo_info).exists():
        default_geo_info = str((base_dir / "examples" / "test_geo_info.json").resolve())
        
    default_output_dir = str((base_dir / "data" / "processed").resolve())
    
    parser.add_argument("--image", default=default_image, help="Input inversion image path")
    parser.add_argument("--geo-info", default=default_geo_info, help="Input geo info JSON path")
    parser.add_argument("--output-dir", default=default_output_dir, help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Workflow I (Image Analysis)
        print("\n=== Step 1: Image Analysis ===")
        # Sub-directory for Step 1
        step1_dir = output_dir / "1_image_analysis"
        step1_dir.mkdir(parents=True, exist_ok=True)
        
        wf1_data = run_image_analyze(args.image, args.geo_info, step1_dir)
        
        # Step 2: Workflow II (Search & Score)
        print("\n=== Step 2: Literature Search & Scoring ===")
        # Sub-directory for Step 2
        step2_dir = output_dir / "2_literature_search"
        step2_dir.mkdir(parents=True, exist_ok=True)
        
        scored_papers = run_literature_search(wf1_data, step2_dir)
        
        # Step 3: Workflow III (Download)
        print("\n=== Step 3: Paper Download ===")
        # Sub-directory for Step 3
        step3_dir = output_dir / "3_paper_download"
        step3_dir.mkdir(parents=True, exist_ok=True)
        
        await run_paper_download(scored_papers, step3_dir)
        
        print("\n>>> All Workflows Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Workflow Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
