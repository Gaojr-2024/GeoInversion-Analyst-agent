import argparse
import json
import sys
import os
import re
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------
# 路径配置：确保无论从哪里运行，都能找到 langchain_version 包
# ----------------------------------------------------------------------
current_file = Path(__file__).resolve()
current_dir = current_file.parent # agents
project_root = current_dir.parent  # langchain_version

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Fix imports for config
try:
    from config import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, TEXT_MODEL, VISION_MODEL
except ImportError:
    try:
        from ..config import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, TEXT_MODEL, VISION_MODEL
    except ImportError:
        # Last resort
        sys.path.append(str(project_root))
        from config import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, TEXT_MODEL, VISION_MODEL

# Import Literature Agent Components
try:
    from agents.literature_agent_v2.search_engine import OpenAlexSearcher
    from agents.literature_agent_v2.query_generator import QueryGenerator
    from agents.literature_agent_v2.relevance_scorer import RelevanceScorer
except ImportError:
    from literature_agent_v2.search_engine import OpenAlexSearcher
    from literature_agent_v2.query_generator import QueryGenerator
    from literature_agent_v2.relevance_scorer import RelevanceScorer

# ==========================================
# Data Structures
# ==========================================

@dataclass
class PaperInfo:
    id: str
    title: str
    score: int
    oa_status: str
    link: str
    year: str = ""
    author: str = ""
    journal: str = ""
    cited_by_count: int = 0
    abstract: str = ""  # Added for report generation
    reason: str = ""    # Added for report generation
    download_status: str = "Pending" # Pending, Success, Failed, Skipped
    local_path: str = ""
    fail_reason: str = ""

# ==========================================
# Helper Classes
# ==========================================

# Import static journal list
try:
    from agents.literature_agent_v2.search_engine import TOP_JOURNALS
except ImportError:
    from literature_agent_v2.search_engine import TOP_JOURNALS

class JournalRanker:
    def __init__(self, output_dir: Path):
        self.llm = ChatOpenAI(
            model=TEXT_MODEL,
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            temperature=0.0
        )
        self.output_dir = output_dir
        self.cache_file = output_dir / "journal_tier_cache.json"
        self.cache = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
                
        # --- Pre-fill Cache with Static KG (TOP_JOURNALS) ---
        # This reduces LLM hallucination for known journals
        self._init_static_knowledge()

    def _init_static_knowledge(self):
        """Inject static knowledge graph into cache to reduce LLM variance."""
        # Tier 1: Nature/Science
        self.cache["Nature"] = 1
        self.cache["Science"] = 1
        
        # Tier 2: Top Sub-journals
        self.cache["Nature Geoscience"] = 2
        self.cache["Science Advances"] = 2
        self.cache["Nature Communications"] = 2
        self.cache["PNAS"] = 2 # Usually Tier 2 equivalent
        
        # Tier 3: Known Disciplinary Top (from search_engine.py)
        # In search_engine.py, TOP_JOURNALS contains mix of Tier 2 and Tier 3.
        # We can map them generally to Tier 3 unless specified above.
        for name in TOP_JOURNALS.keys():
            if name not in self.cache:
                self.cache[name] = 3

    def suggest_journals(self, topic: str) -> Dict[int, List[str]]:
        """
        Semi-RAG Approach:
        1. Retrieve candidates from static Knowledge Graph (TOP_JOURNALS) relevant to topic.
        2. Ask LLM to filtering/adding based on specific topic.
        """
        print(f"  [Ranker] 正在分析主题 '{topic}' 并生成目标期刊列表 (Hybrid RAG)...")
        
        # Static Candidates (Context)
        static_candidates = list(TOP_JOURNALS.keys())
        static_context = ", ".join(static_candidates[:50]) # Limit length

        prompt = f"""
        Research Topic: {topic}
        
        I have a database of high-quality journals:
        {static_context}
        
        Please select the most relevant journals from this list for the topic, and add other missing top-tier journals if necessary.
        Categorize them into Tiers.
        
        Tier 1: Top Multidisciplinary (Nature, Science ONLY)
        Tier 2: Top Field-Specific (Nature Geoscience, Science Advances, Nature Communications, PNAS)
        Tier 3: Reputable Disciplinary (e.g., JGR, GRL, EPSL, Tectonophysics, Geophysics, GJI)
        
        Requirements:
        1. PRIORITIZE selecting from the provided database list.
        2. Tier 1: Max 2 journals.
        3. Tier 2: Max 5 journals.
        4. Tier 3: Max 15 journals.
        5. Return JSON ONLY.
        
        Example JSON:
        {{
            "1": ["Nature", "Science"],
            "2": ["Nature Geoscience"],
            "3": ["Journal of Geophysical Research: Solid Earth", "Earth and Planetary Science Letters"]
        }}
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            # Clean JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            # Normalize and Validate
            result = {1: [], 2: [], 3: []}
            
            # Tier 1 Hard Enforcement
            result[1] = ["Nature", "Science"]
            
            # Process LLM output
            for k, v in data.items():
                try:
                    tier = int(k)
                    if tier == 1: continue # Skip LLM Tier 1, use hardcoded
                    if tier in result and isinstance(v, list):
                        result[tier] = [str(x).strip() for x in v if x]
                except:
                    pass
            
            # Fallback/Enforcement for Tier 2/3 if empty
            if not result[2]:
                result[2] = ["Nature Geoscience", "Science Advances", "Nature Communications"]
            
            # Ensure static top journals are present if relevant (Simple keyword match fallback)
            # This is a simple "Retrieval" check
            keywords = topic.lower().split()
            if "seismic" in keywords or "tomography" in keywords:
                if "Journal of Geophysical Research: Solid Earth" not in result[3]:
                    result[3].append("Journal of Geophysical Research: Solid Earth")
                if "Geophysical Research Letters" not in result[3]:
                    result[3].append("Geophysical Research Letters")
            
            return result
        except Exception as e:
            print(f"  [Ranker] Generate journals failed: {e}")
            # Fallback to static list
            return {
                1: ["Nature", "Science"], 
                2: ["Nature Geoscience", "Science Advances"], 
                3: ["Journal of Geophysical Research: Solid Earth", "Earth and Planetary Science Letters", "Geophysical Research Letters"]
            }

    def classify_tier(self, journal: str) -> int:
        if not journal:
            return 6
        journal = journal.strip()
        if journal in self.cache:
            return self.cache[journal]
        
        # 快速规则匹配
        j_upper = journal.upper()
        if j_upper in ["NATURE", "SCIENCE"]:
            return 1
        
        try:
            response = self.llm.invoke([HumanMessage(content=self.tier_prompt.format(journal=journal))])
            content = response.content.strip()
            match = re.search(r'\d', content)
            if match:
                tier = int(match.group(0))
            else:
                tier = 6
        except Exception as e:
            print(f"  [Ranker] Error classifying '{journal}': {e}")
            tier = 6
        
        self.cache[journal] = tier
        return tier
        
    def save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [Warning] 无法保存期刊缓存: {e}")

# ==========================================
# Main Search & Score Workflow (No Download)
# ==========================================

def run_search_and_score(wf1_data: Dict[str, Any], output_dir: Path) -> List[PaperInfo]:
    """
    Workflow II: Search & Score (Strictly following download.py logic)
    """
    print(f"\n>>> [Workflow II] Starting Search & Score (Staged Search)...")
    
    geo_info = wf1_data["geo_info"]
    image_analysis = wf1_data["image_analysis"]
    wf1_report = wf1_data["wf1_report"]
    
    wf1_bundle = {
        "geo_info": geo_info,
        "image_analysis": image_analysis,
        "initial_explain": wf1_report
    }
    
    # 1. Generate Queries
    print(">>> [Workflow II] Generating Search Queries...")
    query_gen = QueryGenerator() 
    query_pack = query_gen.generate_queries(wf1_bundle)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "query_pack.json", "w", encoding="utf-8") as f:
        json.dump(query_pack, f, ensure_ascii=False, indent=2)
        
    queries = query_pack.get("queries", [])
    negative_terms = query_pack.get("negative_terms", [])
    print(f"    Generated {len(queries)} queries: {queries}")
    
    # 2. 统一检索 (Unified Search with High Quality Filter)
    print(">>> [Workflow II] Executing Unified Search (Relevance + Quality)...")
    searcher = OpenAlexSearcher()
    ranker = JournalRanker(output_dir)
    
    # 2.1 获取高质量期刊列表作为 Filter 或 Boosting
    # 我们不再按 Tier 强制分批检索，而是获取一个大的高质量期刊集合用于过滤/优先
    topic_hint = queries[0] if queries else "Earth Science"
    target_journals_dict = ranker.suggest_journals(topic_hint)
    
    # 合并所有高质量期刊
    high_quality_journals = []
    for tier in [1, 2, 3]:
        high_quality_journals.extend(target_journals_dict.get(tier, []))
    
    print(f"    Target High Quality Journals ({len(high_quality_journals)}): {high_quality_journals[:5]}...")
    
    all_candidates = {} # ID -> Paper
    total_limit = 200 # 增加初筛数量
    
    # Cache for Source IDs
    source_id_cache_file = output_dir / "source_id_cache.json"
    source_id_cache = {}
    if source_id_cache_file.exists():
        try:
             with open(source_id_cache_file, "r", encoding="utf-8") as f:
                 source_id_cache = json.load(f)
        except: pass

    def resolve_ids(journals):
        ids = []
        changed = False
        for j in journals:
            if j in source_id_cache:
                if source_id_cache[j]: ids.append(source_id_cache[j])
            else:
                # print(f"      Resolving ID: {j} ...") # Reduce noise
                sid = searcher.get_source_id(j)
                source_id_cache[j] = sid
                if sid: ids.append(sid)
                changed = True
                # time.sleep(0.1) # Rate limit
        
        if changed:
            with open(source_id_cache_file, "w", encoding="utf-8") as f:
                json.dump(source_id_cache, f, indent=2)
        return ids
    
    hq_source_ids = resolve_ids(high_quality_journals)
    print(f"    Resolved {len(hq_source_ids)} Source IDs for High Quality Journals.")

    # 2.2 执行检索
    # 策略：
    # 1. 优先检索“高质量期刊 + 关键词” (High Precision)
    # 2. 补充检索“关键词” (High Recall)，但后续排序会利用 ranker 降权低质量期刊
    
    # A. 高质量期刊限定检索 (High Precision)
    if hq_source_ids:
        print(f"    [Search Phase A] Searching within High Quality Journals...")
        # Chunking source IDs
        chunk_size = 20
        id_chunks = [hq_source_ids[i:i + chunk_size] for i in range(0, len(hq_source_ids), chunk_size)]
        
        # 限制 Phase A 的数量，避免全被同一两个期刊占满，但又要保证足够多
        phase_a_limit = 100 
        
        for chunk in id_chunks:
            if len(all_candidates) >= phase_a_limit:
                break
            filter_str = "|".join(chunk)
            res = searcher.search(queries, negative_terms=negative_terms, limit_per_query=30, 
                                extra_filters={"primary_location.source.id": filter_str})
            for r in res:
                if r['id'] not in all_candidates:
                    all_candidates[r['id']] = r
                    all_candidates[r['id']]['is_hq_search'] = True
    
    print(f"      -> Found {len(all_candidates)} HQ papers.")
    
    # B. 广义检索 (Broad Search) - 补充剩余名额
    if len(all_candidates) < total_limit:
        needed = total_limit - len(all_candidates)
        print(f"    [Search Phase B] Broad Search for remaining {needed} papers...")
        # 稍微放宽 limit_per_query 以获取更多候选
        res = searcher.search(queries, negative_terms=negative_terms, limit_per_query=max(20, int(needed/len(queries)) + 5))
        
        new_count = 0
        for r in res:
            if r['id'] not in all_candidates:
                all_candidates[r['id']] = r
                all_candidates[r['id']]['is_hq_search'] = False
                new_count += 1
            if len(all_candidates) >= total_limit:
                break
        print(f"      -> Added {new_count} broad papers.")

    raw_candidates = list(all_candidates.values())
    print(f"    Total Retrieved Candidates: {len(raw_candidates)}")
    
    # 3. Intelligent Sorting (Relevance Only)
    print(">>> [Workflow II] Prioritizing Candidates (Relevance Based)...")
    
    # OpenAlex 默认返回就是按相关性排序的，所以我们主要依赖 API 的原始顺序
    # 但由于我们分了两步检索 (HQ + Broad)，我们需要简单合并
    # 如果希望能保持简单的“相关性优先”，其实直接做一次 Broad Search 就够了
    # 不过既然已经做了 HQ 优先，我们可以简单地把 HQ 的放在前面，或者干脆重新按某个简单指标排一下
    # 用户要求：简单一点就好，就简单根据关键词相关性检索
    
    # 既然用户强调“简单根据关键词相关性检索”，那最简单的做法其实是：
    # 信任 Search Engine 返回的顺序。
    # 由于我们是分批拿回来的，这里做一个简单的去重后列表即可。
    # OpenAlex 的 score 字段并不总是对外暴露明确的 relevance score，
    # 但返回顺序通常是可靠的。
    
    candidates = raw_candidates[:200] # Take top 200 based on retrieval order (which roughly correlates to relevance/priority logic we set)
    print(f"    Selected Top {len(candidates)} candidates for scoring.")
    
    # Generate Pre-Score Report (Removed to reduce noise)
    # pre_score_report = output_dir / "retrieved_candidates.md"
    # ...
            
    # Save raw candidates (Removed to reduce noise)
    # with open(output_dir / "search_results_raw.jsonl", "w", encoding="utf-8") as f:
    #     for c in candidates:
    #         f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 4. Score Papers
    print(">>> [Workflow II] Scoring Papers (LLM)...")
    scorer = RelevanceScorer()
    scored_papers_dicts = scorer.score_papers(wf1_bundle, candidates)
    
    # Convert to PaperInfo objects
    paper_objects = []
    for p in scored_papers_dicts:
        journal = p.get("venue", "")
        authors = p.get("authors", [])
        if isinstance(authors, list):
            author_str = ", ".join(authors[:3])
        else:
            author_str = str(authors)

        paper_obj = PaperInfo(
            id=str(p.get("id", "")),
            title=p.get("title", "Untitled"),
            score=int(p.get("score", 0)),
            oa_status="OA" if p.get("is_oa") else "Closed",
            link=p.get("link") or p.get("pdf_url") or p.get("landing_page_url") or "",
            year=str(p.get("year", "")),
            author=author_str,
            journal=journal,
            cited_by_count=p.get("cited_by_count", 0),
            abstract=p.get("abstract", ""),
            reason=p.get("reason", "")
        )
        paper_objects.append(paper_obj)
        
    # Sort by score
    paper_objects.sort(key=lambda x: x.score, reverse=True)
    
    # Save scored list (Removed to reduce noise)
    # with open(output_dir / "candidates_scored.jsonl", "w", encoding="utf-8") as f:
    #     for p in scored_papers_dicts:
    #         f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    # Generate Scored Report
    report_path = output_dir / "literature_review_scored.md"
    _generate_scored_report(paper_objects, report_path)
    
    print(f">>> [Workflow II] Completed. Scored report saved to {report_path}")
    return paper_objects

def _generate_scored_report(papers: List[PaperInfo], output_path: Path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 文献检索与评分报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计检索: {len(papers)} 篇\n\n")
        
        high_score = [p for p in papers if p.score >= 60]
        f.write(f"## 🌟 高分文献 (Score >= 60) - {len(high_score)} 篇\n")
        f.write("> 这些文献将被送入下载流程。\n\n")
        
        for i, p in enumerate(high_score, 1):
            f.write(f"### {i}. {p.title}\n")
            f.write(f"- **评分**: {p.score}\n")
            f.write(f"- **理由**: {p.reason}\n")
            f.write(f"- **来源**: {p.journal} ({p.year}) Cited: {p.cited_by_count}\n")
            f.write(f"- **链接**: [{p.link}]({p.link})\n\n")
            
        f.write("---\n\n")
        
        low_score = [p for p in papers if p.score < 60]
        f.write(f"## 📉 低分文献 (Score < 60) - {len(low_score)} 篇\n")
        for i, p in enumerate(low_score, 1):
            f.write(f"{i}. [{p.score}] {p.title} - {p.reason}\n")

# Main entry for standalone usage
async def main():
    base_dir = Path(__file__).resolve().parent
    default_geo_info = str((base_dir / "examples" / "test_geo_info.json").resolve())
    default_image_analysis = str((base_dir / "examples" / "image_analysis.json").resolve())
    default_wf1_report = str((base_dir / "examples" / "out_report.md").resolve())
    default_output_dir = str((base_dir / "examples" / "download_test_merged").resolve())

    parser = argparse.ArgumentParser(description="Workflow II: 检索与评分 (无下载)")
    parser.add_argument("--geo-info", default=default_geo_info, help="地理信息JSON文件路径")
    parser.add_argument("--image-analysis", default=default_image_analysis, help="image_analysis.json 路径")
    parser.add_argument("--wf1-report", default=default_wf1_report, help="out_report.md 路径")
    parser.add_argument("--output-dir", default=default_output_dir, help="输出目录")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    
    print(f">>> 启动 Workflow II (检索 -> 评分)...")
    
    if not Path(args.geo_info).exists():
        print(f"[Error] geo_info not found: {args.geo_info}")
        return
        
    geo_info = json.loads(Path(args.geo_info).read_text(encoding="utf-8"))
    image_analysis = json.loads(Path(args.image_analysis).read_text(encoding="utf-8"))
    wf1_report = Path(args.wf1_report).read_text(encoding="utf-8")
    
    wf1_data = {
        "geo_info": geo_info,
        "image_analysis": image_analysis,
        "wf1_report": wf1_report
    }

    run_search_and_score(wf1_data, output_dir)

if __name__ == "__main__":
    # download.py now acts as search_and_score.py
    asyncio.run(main())
