import os
import json
from pathlib import Path
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import Config from core
from app.core.config import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, TEXT_MODEL, get_prompt_path

class ReportGenerationChain:
    """
    使用 LangChain 实现的报告生成器
    采用 Map-Reduce 思路：
    1. Map: 独立分析每篇文献，提取与用户结果的关联
    2. Reduce: 基于提取的关联信息，分章节撰写报告
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 初始化大模型 (The Brain)
        self.llm = ChatOpenAI(
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            model_name=TEXT_MODEL,
            temperature=0.3, # 保持低创造性，高严谨性
            max_tokens=4000  # 单次输出限制
        )
        
        # 2. 初始化 Prompt (The Instructions)
        self._init_prompts()

    def _load_prompt(self, prompt_name: str) -> str:
        """加载外部 Prompt 文件"""
        path = get_prompt_path(prompt_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Critical Error: Prompt file not found at {path}. Please check your configuration.")

    def _init_prompts(self):
        # --- Map Prompt: 单篇文献分析 ---
        map_template = self._load_prompt("map_analysis")
        self.map_prompt = PromptTemplate(
            input_variables=["user_analysis", "paper_title", "paper_summary", "paper_authors", "paper_year", "paper_journal"],
            template=map_template
        )
        
        # --- Reduce Prompt: 章节撰写 ---
        reduce_template = self._load_prompt("reduce_report")
        self.section_prompt = PromptTemplate(
            input_variables=["section_name", "section_requirements", "summaries"],
            template=reduce_template
        )

    def load_user_analysis(self, json_path: str) -> str:
        """加载用户的反演结果分析"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 提取关键信息转换为字符串
        analysis_data = data.get("user_analysis", data)
        return json.dumps(analysis_data, indent=2, ensure_ascii=False)

    def build_literature_docs(self, mineru_output_dir: str) -> List[Dict]:
        """
        加载所有文献数据，构建文档列表
        相当于 LangChain 的 DocumentLoader
        """
        papers = []
        base_path = Path(mineru_output_dir)
        
        for paper_dir in base_path.iterdir():
            if not paper_dir.is_dir():
                continue
            
            # Use filtered_analysis.json as per new pipeline logic
            analysis_path = paper_dir / "filtered_analysis.json"
            
            # Fallback to old file if new one doesn't exist (backward compatibility)
            if not analysis_path.exists():
                analysis_path = paper_dir / "image_analysis_v2.json"
                
            if not analysis_path.exists():
                continue
                
            # Load Analysis
            analysis = []
            if analysis_path.exists():
                with open(analysis_path, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
            
            # Load Literature (New)
            lit_path = paper_dir / "literature.json"
            lit_info = {}
            if lit_path.exists():
                with open(lit_path, 'r', encoding='utf-8') as f:
                    lit_info = json.load(f)
            
            # If no analysis (images) AND no literature info, skip
            if not analysis and not lit_info:
                continue
                
            # Metadata is now embedded in the items in filtered_analysis.json
            # But we can take it from the first item if available
            title = paper_dir.name
            authors = "Unknown"
            year = "N/A"
            journal = "N/A"
            journal_rank_str = ""
            
            if analysis and len(analysis) > 0:
                first_item = analysis[0]
                # Try to get info from item
                journal = first_item.get("journal_name", "N/A")
                
                # Format journal rank string
                rank_info = first_item.get("journal_rank", {})
                if rank_info:
                    impact = rank_info.get("impact_factor", "N/A")
                    quartile = rank_info.get("quartile", "N/A")
                    journal_rank_str = f" (IF: {impact}, {quartile})"
                
                # If paper_info is still there (merged), use it
                paper_info = first_item.get("paper_info", {})
                if paper_info:
                    title = paper_info.get("title", title)
                    # Handle authors which might be a list or string
                    auth_raw = paper_info.get("authors", [])
                    if isinstance(auth_raw, list):
                        authors = ", ".join(auth_raw)
                    else:
                        authors = str(auth_raw)
                    year = paper_info.get("year", year)
            
            # If still missing, try paper_info.json (legacy)
            info_path = paper_dir / "paper_info.json"
            # Even if we got some info, paper_info.json might be more complete for authors/year
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    # Prefer meta title if available
                    if meta.get("title"): title = meta.get("title")
                    
                    # Fix authors extraction
                    auth_raw = meta.get("authors", [])
                    if isinstance(auth_raw, list) and auth_raw:
                        authors = ", ".join(auth_raw)
                    elif isinstance(auth_raw, str) and auth_raw:
                        authors = auth_raw
                    
                    if meta.get("year"): year = meta.get("year")
                    if journal == "N/A" and meta.get("journal"):
                        journal = meta.get("journal")

            full_journal_str = f"{journal}{journal_rank_str}"

            # Aggregate findings
            findings = []
            
            # Add Global Summary from Literature.json (Preferred)
            if lit_info:
                findings.append(f"【全文总结】\n背景: {lit_info.get('background','')}\n结论: {lit_info.get('conclusions','')}\n主要观测: {lit_info.get('results','')}")
            # Fallback to analysis item
            elif analysis and len(analysis) > 0 and "paper_summary" in analysis[0]:
                summary = analysis[0]["paper_summary"]
                findings.append(f"【全文总结】\n背景: {summary.get('background','')}\n结论: {summary.get('conclusions','')}")

            for img in analysis:
                # Assuming filtered_analysis only has relevant images
                caption = img.get('caption', '')[:100]
                findings.append(f"\n[图片: {img.get('img_name')}] {caption}...")
                
                # Text Evidence (New Feature)
                if img.get("text_evidence"):
                    findings.append("【原文关键证据】:")
                    for ev in img.get("text_evidence", []):
                        # Handle both string (old) and dict (new)
                        if isinstance(ev, str):
                            findings.append(f"  > {ev}")
                        elif isinstance(ev, dict):
                            content = ev.get("content", "")
                            loc = ev.get("location", "Unknown")
                            findings.append(f"  > \"{content}\" (Location: {loc})")
                
                # Original Context
                if img.get("original_interpretation"):
                     findings.append(f"【上下文提取】: {img.get('original_interpretation')[:300]}...")

                # Model Analysis
                findings.append("【模型分析】:")
                for anomaly in img.get("anomalies", []):
                    findings.append(f"  - 异常: {anomaly.get('description')}")
                    if anomaly.get("interpretation"):
                        findings.append(f"    解释: {anomaly.get('interpretation')}")

            content_str = "\n".join(findings)
            if not content_str:
                content_str = "未提取到有效的反演异常信息。"
                
            papers.append({
                "title": title,
                "authors": authors,
                "year": year,
                "journal": full_journal_str,
                "summary": content_str[:20000] # Increased limit for rich evidence
            })
            
        return papers

    def generate_report(self, user_analysis_json: str, mineru_output_dir: str, output_path: str):
        print("\n>>> 开始 LangChain 报告生成流程 (Map-Reduce 模式) <<<")
        
        # 1. Data Loading
        user_data_str = self.load_user_analysis(user_analysis_json)
        papers = self.build_literature_docs(mineru_output_dir)
        print(f"已加载 {len(papers)} 篇文献作为上下文。")
        
        # 2. Map Phase: Analyze each paper individually
        map_chain = self.map_prompt | self.llm | StrOutputParser()
        
        analyzed_summaries = []
        print("\n--- 阶段 1: 单篇文献交叉分析 (Map) ---")
        
        for i, paper in enumerate(papers):
            print(f"[{i+1}/{len(papers)}] 正在分析: {paper['title'][:50]}...")
            
            # Ensure no "N/A" or "Unknown" passed if possible, or handle in prompt
            # But the prompt expects strings.
            try:
                res = map_chain.invoke({
                    "user_analysis": user_data_str,
                    "paper_title": paper['title'],
                    "paper_authors": paper['authors'],
                    "paper_year": paper['year'],
                    "paper_journal": paper['journal'],
                    "paper_summary": paper['summary']
                })
                analyzed_summaries.append(res)
            except Exception as e:
                print(f"分析失败: {e}")
            
        combined_summaries = "\n\n".join(analyzed_summaries)
        
        # 保存中间结果 (Map Result)
        intermediate_path = Path(output_path).parent / "intermediate_summaries.txt"
        with open(intermediate_path, "w", encoding="utf-8") as f:
            f.write(combined_summaries)
        print(f"中间分析结果已保存: {intermediate_path}")
            
        # 3. Reduce Phase: Generate Report Sections
        print("\n--- 阶段 2: 分章节撰写报告 (Reduce) ---")
        section_chain = self.section_prompt | self.llm | StrOutputParser()
        
        # 定义报告结构 (Outline)
        sections = [
            ("摘要", "简要概括主要发现及文献佐证情况。"),
            ("一、反演结果详述", "描述用户图中识别出的关键异常（位置、形态、物性特征），并结合地质背景进行初步推断。"),
            ("二、文献综合对比与讨论", "这是核心部分。请详细分类讨论：1.一致性分析(佐证)；2.差异性分析(反对)；3.互补性分析。引用文献时请加粗作者和年份。请充分利用提供的文献摘要，特别是【原文关键证据】部分。"),
            ("三、地质成因解释模型", "综合用户结果和文献证据，提出一个最合理的地下结构成因模型（如：板片断离、地幔柱上涌等）。"),
            ("四、结论", "总结核心观点，指出未来研究方向。")
        ]
        
        full_report = "# 地球物理反演结果综合解释报告 (LangChain版)\n\n"
        
        for title, req in sections:
            print(f"正在撰写: {title}...")
            try:
                section_content = section_chain.invoke({
                    "section_name": title,
                    "section_requirements": req,
                    "summaries": combined_summaries
                })
                full_report += f"## {title}\n\n{section_content}\n\n"
            except Exception as e:
                print(f"章节撰写失败: {e}")
                full_report += f"## {title}\n\n(撰写失败)\n\n"
            
        # 4. Save Final Report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_report)
            
        print(f"\n报告生成完成: {output_path}")
