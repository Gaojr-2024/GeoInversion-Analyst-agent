import os
import json
import re
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any

# Import Components
from app.tools.pdf_extractor import PDFFigureExtractor
from app.chains.image_analysis import ImageAnalysisChain
from app.chains.report_generator import ReportGenerationChain
from app.chains.journal_search import JournalSearchChain
from app.chains.literature_extraction import LiteratureExtractionChain
from app.core.config import MAX_CONCURRENCY

class GeoInversionPipeline:
    """
    全流程 LangChain 自动化分析管线
    流程: PDF拆解 -> 全文信息提取 -> 期刊搜索 -> 图片并行语义分析 -> 结果过滤与增强 -> 综合报告生成
    """
    
    def __init__(self, pdf_input_dir: str, extraction_output_dir: str, final_report_path: str):
        self.pdf_input_dir = Path(pdf_input_dir)
        self.extraction_output_dir = Path(extraction_output_dir)
        self.final_report_path = Path(final_report_path)
        
        # 1. PDF Extractor Tool
        self.pdf_extractor = PDFFigureExtractor(output_base_dir=str(extraction_output_dir))
        
        # 2. Chains
        self.image_analyzer = ImageAnalysisChain()
        self.journal_search = JournalSearchChain()
        self.lit_extractor = LiteratureExtractionChain()
        self.report_generator = ReportGenerationChain(output_dir=os.path.dirname(final_report_path))

    def _extract_metadata(self, paper_folder_name: str, full_text: str) -> Dict[str, str]:
        """从文件夹名和全文中提取元数据"""
        metadata = {
            "title": "",
            "authors": [],
            "year": "",
            "journal": ""
        }
        
        # 1. Parse Folder Name
        try:
            parts = paper_folder_name.split(" - ")
            if len(parts) >= 4:
                # Format: Journal - Year - Author - Title
                metadata["journal"] = parts[0].strip()
                metadata["year"] = parts[1].strip()
                metadata["authors"] = [parts[2].replace(" 等", "").strip()]
                metadata["title"] = parts[3].strip()
            elif len(parts) == 3:
                # Format: Author - Year - Title
                metadata["authors"] = [parts[0].replace(" 等", "").strip()]
                metadata["year"] = parts[1].strip()
                metadata["title"] = parts[2].strip()
        except Exception:
            pass

        # 2. Try to find Title in full_text
        if not metadata["title"] and full_text:
            lines = full_text.split('\n')
            for line in lines[:20]:
                if line.startswith("# "):
                    metadata["title"] = line.replace("# ", "").strip()
                    break
        
        return metadata

    def _extract_figure_context(self, full_text: str, figure_label: str) -> str:
        """从全文中提取关于该图片的上下文段落 (基于规则)"""
        if not full_text or not figure_label:
            return ""
            
        label_match = re.search(r'(Figure|Fig\.|FIG\.)\s*(\d+)', figure_label, re.IGNORECASE)
        if not label_match:
            return ""
            
        fig_num = label_match.group(2)
        patterns = [f"Figure {fig_num}", f"Fig. {fig_num}", f"FIG. {fig_num}"]
        
        paragraphs = full_text.split('\n\n')
        relevant_paragraphs = []
        
        for p in paragraphs:
            clean_p = p.strip()
            # Skip caption itself
            if re.match(r'^(\*\*|!\[.*\]\(.*\)[\r\n]*\*\*)?\s*(Figure|Fig\.|FIG\.)\s*' + fig_num, clean_p, re.IGNORECASE):
                continue
                
            for pat in patterns:
                if pat in p:
                    relevant_paragraphs.append(clean_p)
                    break
        
        return "\n\n".join(relevant_paragraphs)

    def _process_single_image(self, item: Dict, paper_dir: Path, full_text: str, metadata: Dict) -> Dict:
        """处理单张图片 (用于并行调用)"""
        img_path_rel = item.get("img_path")
        caption = item.get("caption", "")
        img_name = item.get("img_name")
        
        if not img_path_rel:
            return None
            
        img_path_abs = paper_dir / img_path_rel
        if not img_path_abs.exists():
            return None
        
        # Context Extraction (Rule-based)
        fig_label_match = re.match(r'^(Figure|Fig\.|FIG\.)\s*\d+', caption, re.IGNORECASE)
        fig_label = fig_label_match.group(0) if fig_label_match else ""
        original_interpretation = self._extract_figure_context(full_text, fig_label)
        
        context = {
            "caption": caption,
            "original_interpretation": original_interpretation,
            "paper_info": metadata
        }
        
        print(f"  [Thread] 分析图片: {img_name} ...")
        # Call LangChain Image Analyzer
        try:
            result = self.image_analyzer.analyze_image(str(img_path_abs), context)
        except Exception as e:
            print(f"  [Error] 图片分析失败 {img_name}: {e}")
            result = {"is_inversion_image": False, "error": str(e)}
        
        # Merge result
        result["img_name"] = img_name
        result["caption"] = caption
        result["original_interpretation"] = original_interpretation
        result["fig_label"] = fig_label # Store for later matching
        # Note: metadata is passed in context but not merged here to avoid redundancy, 
        # we will merge consolidated info later.
        
        return result

    def run(self, user_analysis_json: str):
        print(">>> 启动 LangChain 全流程分析管线 <<<")
        print(f"并发设置 (MAX_CONCURRENCY): {MAX_CONCURRENCY}")
        
        # --- Step 1: PDF Extraction ---
        print("\n=== 阶段 1: PDF 拆解与提取 ===")
        pdf_files = list(self.pdf_input_dir.glob("*.pdf"))
        print(f"发现 {len(pdf_files)} 个PDF文件。")
        
        processed_folders = []
        for pdf_path in pdf_files:
            try:
                output_dir = self.pdf_extractor.process_pdf(pdf_path)
                processed_folders.append(output_dir)
            except Exception as e:
                print(f"处理 PDF 失败 {pdf_path.name}: {e}")

        # --- Step 2: Analysis & Enrichment ---
        print("\n=== 阶段 2: 全文提取、期刊搜索与图片并行分析 ===")
        
        for paper_dir in processed_folders:
            print(f"\n正在处理文献: {paper_dir.name}")
            
            # 2.1 Load Content
            content_list_path = paper_dir / "content_list.json"
            full_text_path = paper_dir / "full.md"
            
            if not content_list_path.exists():
                continue
                
            with open(content_list_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
                
            full_text = ""
            if full_text_path.exists():
                with open(full_text_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
            
            # 2.2 Extract Metadata
            metadata = self._extract_metadata(paper_dir.name, full_text)
            
            # 2.3 Journal Search (if needed)
            journal_rank_info = {}
            if not metadata.get("journal") or "quality_info" not in metadata:
                print(f"  - 补充期刊质量信息 (LetPub/Web): {metadata.get('title', 'Unknown')[:30]}...")
                try:
                    journal_rank_info = self.journal_search.search_journal_info(metadata.get("title", ""))
                    if not metadata.get("journal") and journal_rank_info.get("journal_name") != "Unknown":
                        metadata["journal"] = journal_rank_info.get("journal_name")
                except Exception as e:
                    print(f"    期刊信息搜索失败: {e}")
            
            # Save Metadata (Restored)
            # This is crucial for report generator fallback and general record keeping
            with open(paper_dir / "paper_info.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # 2.4 Literature Extraction (Full Text Evidence)
            print("  - 提取全文关键信息与图表证据...")
            lit_info = {}
            try:
                lit_info = self.lit_extractor.extract_info(full_text)
                
                # Merge LLM-extracted metadata if existing metadata is incomplete
                llm_meta = lit_info.get("metadata", {})
                if llm_meta:
                    print(f"    [LLM Metadata] Title: {llm_meta.get('title')[:30]}..., Authors: {llm_meta.get('authors')}, Year: {llm_meta.get('year')}")
                    
                    if not metadata["title"] and llm_meta.get("title"):
                        metadata["title"] = llm_meta.get("title")
                    
                    if not metadata["authors"] and llm_meta.get("authors"):
                        metadata["authors"] = llm_meta.get("authors")
                        
                    if not metadata["year"] and llm_meta.get("year"):
                        metadata["year"] = str(llm_meta.get("year"))
                        
                    if not metadata["journal"] and llm_meta.get("journal"):
                        metadata["journal"] = llm_meta.get("journal")
                        
                    # Re-save updated metadata
                    with open(paper_dir / "paper_info.json", "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
            except Exception as e:
                print(f"    全文提取失败: {e}")
                
            # Save literature info separately
            with open(paper_dir / "literature.json", "w", encoding="utf-8") as f:
                json.dump(lit_info, f, indent=2, ensure_ascii=False)
                
            figure_evidence_map = lit_info.get("figure_evidence", {})

            # 2.5 Parallel Image Analysis
            raw_analysis_results = []
            output_path_v2 = paper_dir / "image_analysis_v2.json"
            
            # Check cache for raw analysis
            if output_path_v2.exists():
                print(f"  - 加载已有图片分析结果: {output_path_v2.name}")
                with open(output_path_v2, 'r', encoding='utf-8') as f:
                    raw_analysis_results = json.load(f)
            else:
                print(f"  - 并行分析 {len(content_list)} 张图片 (Max Workers: {MAX_CONCURRENCY})...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
                    futures = [
                        executor.submit(self._process_single_image, item, paper_dir, full_text, metadata)
                        for item in content_list
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res:
                            raw_analysis_results.append(res)
                
                # Save raw results (optional, but good for caching)
                with open(output_path_v2, "w", encoding="utf-8") as f:
                    json.dump(raw_analysis_results, f, indent=2, ensure_ascii=False)

            # 2.6 Filter & Enrich (Create New JSON)
            print("  - 过滤与增强结果 (生成 filtered_analysis.json)...")
            filtered_results = []
            
            for item in raw_analysis_results:
                # Filter: Keep only inversion images
                if not item.get("is_inversion_image"):
                    continue
                
                # Enrich 1: Journal Rank
                # Merge the journal info into each item so downstream can use it easily
                item["journal_rank"] = journal_rank_info if journal_rank_info else metadata.get("quality_info", {})
                item["journal_name"] = metadata.get("journal", "Unknown")
                
                # Enrich 1.1: Full Metadata (Authors, Year) - FIX for Unknown (N/A)
                # Ensure we pass the extracted metadata into the item
                item["paper_info"] = metadata

                # Enrich 2: Text Evidence (from Literature Extraction)
                # Try to match figure label
                fig_label = item.get("fig_label", "")
                if not fig_label:
                    # Try to extract again if missing
                    caption = item.get("caption", "")
                    m = re.match(r'^(Figure|Fig\.|FIG\.)\s*\d+', caption, re.IGNORECASE)
                    if m:
                        fig_label = m.group(0)
                
                # Normalize label for lookup (e.g. "Figure 1" vs "Fig. 1")
                # Simple normalization: extract number
                evidence = []
                if fig_label:
                    num_match = re.search(r'\d+', fig_label)
                    if num_match:
                        num = num_match.group(0)
                        # Look for keys containing this number in figure_evidence_map
                        for key, evidence_list in figure_evidence_map.items():
                            if num in key:
                                # New structure: evidence_list is a list of dicts or strings
                                # Handle both for backward compatibility
                                for ev in evidence_list:
                                    if isinstance(ev, str):
                                        evidence.append({"content": ev, "location": "Unknown"})
                                    elif isinstance(ev, dict):
                                        evidence.append(ev)
                
                item["text_evidence"] = evidence
                
                # Enrich 3: Global Literature Info (Optional, but useful)
                item["paper_summary"] = {
                    "background": lit_info.get("background", ""),
                    "methods": lit_info.get("methods", ""),
                    "conclusions": lit_info.get("conclusions", "")
                }
                
                filtered_results.append(item)
            
            # Save Filtered JSON
            filtered_output_path = paper_dir / "filtered_analysis.json"
            with open(filtered_output_path, "w", encoding="utf-8") as f:
                json.dump(filtered_results, f, indent=2, ensure_ascii=False)
            
            print(f"  - 有效反演图片: {len(filtered_results)} 张 (已保存至 {filtered_output_path.name})")

        # --- Step 3: Report Generation ---
        print("\n=== 阶段 3: 综合报告生成 (LangChain Map-Reduce) ===")
        
        try:
            self.report_generator.generate_report(
                user_analysis_json=user_analysis_json,
                mineru_output_dir=str(self.extraction_output_dir),
                output_path=str(self.final_report_path)
            )
            print(f"\nSUCCESS: 全流程执行完毕！报告已生成: {self.final_report_path}")
        except Exception as e:
            print(f"报告生成失败: {e}")
            import traceback
            traceback.print_exc()
