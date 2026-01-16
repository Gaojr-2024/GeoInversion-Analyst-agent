"""
地球物理反演解释Agent - 主应用入口
"""

import json
import argparse
import os
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

try:
    import openai
except Exception:
    openai = None

try:
    # Try relative imports (when run as package)
    from .input_processor import InputProcessor
    from ..config import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, PROMPT_TEMPLATES, TEXT_MODEL
except (ImportError, ValueError):
    # Fallback for direct execution or script usage
    import sys
    from pathlib import Path

    current_dir = Path(__file__).resolve().parent # agents
    project_root = current_dir.parent # langchain_version
    
    # Ensure project root is in path to find config
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Ensure agents dir is in path to find siblings
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from input_processor import InputProcessor
    from config import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, PROMPT_TEMPLATES, TEXT_MODEL

class EarthPhysicsAgent:
    """地球物理反演解释Agent主类"""
    
    def __init__(self, enable_literature: bool = False):
        """初始化所有组件"""
        self.analyzer: Optional[InputProcessor] = None
        self.text_model: Optional[ChatOpenAI] = None

        if ALIBABA_API_KEY:
            self.analyzer = InputProcessor()
            self.text_model = ChatOpenAI(
                model=TEXT_MODEL,
                api_key=ALIBABA_API_KEY,
                base_url=OPENAI_COMPATIBLE_BASE_URL,
                temperature=0.2,
                max_tokens=10000,
            )


    def _read_prompt_template(self, key: str) -> str:
        prompt_path = PROMPT_TEMPLATES.get(key)
        if not prompt_path:
            raise ValueError(f"未知提示词模板key: {key}")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"提示词模板文件不存在: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_json_from_text(self, text: str) -> Any:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                pass

        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError("无法从响应中提取有效的JSON")

    def run_workflow_i(
        self,
        image_path: str,
        geo_info: Dict[str, Any],
        branches: int = 3,
    ) -> Dict[str, Any]:
        if self.analyzer is None:
            raise ValueError("缺少阿里云 API Key：请在 config.py/.env/环境变量中配置")
        image_analysis = self.analyzer.analyze_inversion_image_with_vision_model(image_path, geo_info)
        search_query = self._build_search_query(geo_info, image_analysis)
        initial_interpretation = self._generate_initial_interpretation(geo_info, image_analysis)
        hypotheses = self._generate_hypotheses(geo_info, image_analysis, initial_interpretation, branches=branches)
        evaluation = self._evaluate_hypotheses(geo_info, image_analysis, hypotheses)
        final_report = self._generate_report_workflow_i(
            geo_info=geo_info,
            image_analysis=image_analysis,
            hypotheses=hypotheses,
            evaluation=evaluation,
            initial_interpretation=initial_interpretation,
        )

        return {
            "workflow": "I",
            "image_analysis": image_analysis,
            "initial_interpretation": initial_interpretation,
            "search_query": search_query,
            "hypotheses": hypotheses,
            "evaluation": evaluation,
            "final_report": final_report,
        }

    def run_workflow_ii(
        self,
        image_path: str,
        geo_info: Dict[str, Any],
        download_limit: int = 40,
        literature_k: int = 5,
        artifacts_dir: Optional[str] = None,
        pdf_source: str = "pubmed",
        download_threads: int = 3,
        download_delay: Optional[float] = None,
        wf1_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base_dir = Path(artifacts_dir).resolve() if artifacts_dir else Path(__file__).resolve().parent / "examples"
        base_dir.mkdir(parents=True, exist_ok=True)
        downloads_dir = (base_dir / "downloads").resolve()
        mineru_results_dir = (base_dir / "mineru_results").resolve()
        vector_store_dir = (base_dir / "vector_store").resolve()
        downloads_dir.mkdir(parents=True, exist_ok=True)

        if self.enable_literature:
            if self.retriever is None or str(self.retriever.literature_dir.resolve()) != str(downloads_dir):
                self.retriever = LiteratureRetriever(
                    literature_dir=str(downloads_dir),
                    vector_store_path=str(vector_store_dir),
                    mineru_results_dir=str(mineru_results_dir),
                )

        if wf1_result and isinstance(wf1_result, dict):
            image_analysis = wf1_result.get("image_analysis") or {}
            initial_explain = wf1_result.get("initial_interpretation") or wf1_result.get("initial_explain") or ""
            wf1_bundle = self._build_wf1_bundle(geo_info, image_analysis, str(initial_explain))
            search_query = wf1_result.get("search_query") or self._build_search_query(geo_info, image_analysis)
        else:
            image_analysis = self.analyzer.analyze_inversion_image_with_vision_model(image_path, geo_info)
            initial_explain = self._generate_initial_interpretation(geo_info, image_analysis)
            wf1_bundle = self._build_wf1_bundle(geo_info, image_analysis, initial_explain)
            search_query = self._build_search_query(geo_info, image_analysis)

        query_pack = self._plan_queries(wf1_bundle)
        query_pack_path = base_dir / "query_pack.json"
        with open(query_pack_path, "w", encoding="utf-8") as f:
            json.dump(query_pack, f, ensure_ascii=False, indent=2)

        MCPRetriever = self._import_pdfget()
        mcp = MCPRetriever()

        constraints = query_pack.get("constraints") or {}
        budget = query_pack.get("budget") or {}
        per_provider_topk = int(budget.get("per_provider_topk") or 50)
        max_candidates = int(budget.get("max_candidates") or 300)
        access_topk = int(budget.get("access_topk") or 200)
        download_topk = int(budget.get("download_topk") or download_limit)
        download_topk = min(download_topk, int(download_limit))

        year_from = constraints.get("year_from")
        year_to = constraints.get("year_to")
        year_from = int(year_from) if isinstance(year_from, int) or (isinstance(year_from, str) and str(year_from).isdigit()) else None
        year_to = int(year_to) if isinstance(year_to, int) or (isinstance(year_to, str) and str(year_to).isdigit()) else None

        queries = query_pack.get("queries") or []
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        candidates = mcp.discover(
            queries,
            per_provider_topk=per_provider_topk,
            year_from=year_from,
            year_to=year_to,
            max_workers=max(1, int(download_threads)),
        )

        candidates.sort(key=lambda x: (x.get("cited_by_count", 0), bool(x.get("abstract"))), reverse=True)
        candidates = candidates[: max(1, max_candidates)]

        ranked = self._rerank_candidates(wf1_bundle, candidates)
        papers_ranked_path = base_dir / "papers_ranked.jsonl"
        self._write_jsonl(papers_ranked_path, ranked)

        ranked_with_access = mcp.resolve_access_batch(ranked, topk=access_topk)
        ranked_with_access.sort(key=lambda x: x.get("score", 0), reverse=True)

        download_results = mcp.download_oa_pdfs(
            ranked_with_access,
            out_dir=str(downloads_dir),
            download_topk=download_topk,
            max_workers=max(1, int(download_threads)),
        )

        pid_to_download = {r.get("paper_id"): r for r in download_results if r.get("paper_id")}
        manifest_records: List[Dict[str, Any]] = []
        attempted = 0
        for p in ranked_with_access[: max(1, access_topk)]:
            pid = p.get("paper_id")
            oa_status = p.get("oa_status") or "unknown"
            best_pdf_url = p.get("best_pdf_url") or ""
            rec: Dict[str, Any] = {
                "paper_id": pid,
                "title": p.get("title"),
                "doi": p.get("doi"),
                "pdf_url": best_pdf_url,
                "status": "fail",
                "pdf_path": "",
                "fail_reason": "",
            }
            if oa_status != "oa_pdf" or not best_pdf_url:
                rec["fail_reason"] = oa_status
                manifest_records.append(rec)
                continue
            if attempted >= download_topk:
                rec["fail_reason"] = "budget_skip"
                manifest_records.append(rec)
                continue
            attempted += 1
            r = pid_to_download.get(pid, {})
            if r.get("success"):
                rec["status"] = "success"
                rec["pdf_path"] = r.get("path") or ""
            else:
                rec["status"] = "fail"
                rec["fail_reason"] = r.get("fail_reason") or r.get("error") or "download_failed"
            manifest_records.append(rec)

        manifest_path = base_dir / "download_manifest.jsonl"
        self._write_jsonl(manifest_path, manifest_records)

        literature_results = []
        image_analysis_results = []
        if self.retriever is not None:
            self.retriever.process_pdfs_with_mineru()
            self.retriever.update_vector_store()
            literature_results = self.retriever.retrieve_relevant_literature(search_query, k=literature_k)
            image_analysis_results = self.retriever.analyze_literature_images(literature_results, query_context=search_query)

        report_prompt = (
            "你是一位专业地球物理学家。你将收到：反演结果结构化JSON、WorkflowⅠ初步解释、检索式、排序后的候选论文、以及从本地PDF切片检索到的证据片段。\n"
            "请输出一份Workflow II报告，要求：\n"
            "1) 不强行突破付费墙；closed-access 只给 landing URL/DOI\n"
            "2) 只使用给定 evidence 片段作为证据，不要编造引用\n"
            "3) 用Markdown输出，包含：检索式、下载统计、证据要点(带来源文件名)、与反演异常的对应关系、下一步检索建议\n"
            "\n"
            "wf1_bundle.json：\n"
            "```json\n"
            f"{json.dumps(wf1_bundle, ensure_ascii=False, indent=2)}\n"
            "```\n"
            "\n"
            "query_pack.json：\n"
            "```json\n"
            f"{json.dumps(query_pack, ensure_ascii=False, indent=2)}\n"
            "```\n"
            "\n"
            "下载清单摘要（JSON，前50条）：\n"
            "```json\n"
            f"{json.dumps(manifest_records[:50], ensure_ascii=False, indent=2)}\n"
            "```\n"
            "\n"
            "证据片段（JSON，source为PDF路径）：\n"
            "```json\n"
            f"{json.dumps(literature_results, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
        if self.text_model is None:
            final_report = self._fallback_workflow_ii_report(
                wf1_bundle=wf1_bundle,
                query_pack=query_pack,
                manifest_records=manifest_records,
                literature_results=literature_results,
            )
        else:
            try:
                response = self.text_model.invoke([HumanMessage(content=report_prompt)])
                final_report = str(response.content).strip()
            except Exception:
                final_report = self._fallback_workflow_ii_report(
                    wf1_bundle=wf1_bundle,
                    query_pack=query_pack,
                    manifest_records=manifest_records,
                    literature_results=literature_results,
                )

        return {
            "workflow": "II",
            "artifacts_dir": str(base_dir),
            "image_analysis": image_analysis,
            "initial_explain": initial_explain,
            "wf1_bundle": wf1_bundle,
            "query_pack": query_pack,
            "papers_ranked_path": str(papers_ranked_path),
            "download_manifest_path": str(manifest_path),
            "downloads_dir": str(downloads_dir),
            "literature_results": literature_results,
            "literature_image_analysis_results": image_analysis_results,
            "final_report": final_report,
        }

    def _fallback_workflow_ii_report(
        self,
        wf1_bundle: Dict[str, Any],
        query_pack: Dict[str, Any],
        manifest_records: List[Dict[str, Any]],
        literature_results: Any,
    ) -> str:
        queries = query_pack.get("queries") or []
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        total = len(manifest_records)
        success = sum(1 for r in manifest_records if r.get("status") == "success")
        failed = total - success
        fail_reasons: Dict[str, int] = {}
        for r in manifest_records:
            if r.get("status") != "success":
                reason = str(r.get("fail_reason") or "unknown")
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        top_fails = sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True)[:8]
        fail_lines = "\n".join([f"- {k}: {v}" for k, v in top_fails]) if top_fails else "- 无"

        evidence_lines: List[str] = []
        try:
            if isinstance(literature_results, list):
                for item in literature_results[:10]:
                    if not isinstance(item, dict):
                        continue
                    src = str(item.get("source") or item.get("metadata", {}).get("source") or "").strip()
                    content = str(item.get("content") or item.get("page_content") or "").strip()
                    content = content.replace("\n", " ").strip()
                    if content:
                        snippet = content[:280]
                        label = src if src else "unknown_source"
                        evidence_lines.append(f"- {label}: {snippet}")
        except Exception:
            evidence_lines = []
        if not evidence_lines:
            evidence_lines = ["- 未检索到证据片段（可能尚未成功下载/切片）"]

        anomaly_ids = []
        for c in wf1_bundle.get("claims") or []:
            if isinstance(c, dict) and c.get("anomaly_id"):
                anomaly_ids.append(str(c["anomaly_id"]))
        anomaly_ids = list(dict.fromkeys(anomaly_ids))[:12]
        anomaly_line = ", ".join(anomaly_ids) if anomaly_ids else "未提供"

        query_lines = "\n".join([f"- {q}" for q in queries[:15]]) if queries else "- 未生成"
        region = str((wf1_bundle.get("geo_info") or {}).get("region_name") or "").strip() or "未提供"

        return (
            "# Workflow II 报告（确定性降级输出）\n\n"
            f"## 检索式\n{query_lines}\n\n"
            "## 下载统计\n"
            f"- 总记录: {total}\n"
            f"- 成功: {success}\n"
            f"- 失败: {failed}\n"
            "### 失败原因分布\n"
            f"{fail_lines}\n\n"
            "## 与反演异常的对应关系（待补充）\n"
            f"- 区域: {region}\n"
            f"- anomaly_id: {anomaly_line}\n\n"
            "## 证据要点（来自本地切片检索）\n"
            + "\n".join(evidence_lines)
            + "\n\n"
            "## 下一步检索建议\n"
            "- 增加区域同义词与构造术语（forearc/slab/mantle wedge 等）组合检索\n"
            "- 若大量 oa_html/closed，降低 access_topk 或提高 openalex/semanticscholar 的召回上限\n"
            "- 若 evidence 为空，优先检查 downloads/ 是否有 PDF，以及 MinerU 是否完成切片\n"
        )

    def _build_search_query(self, geo_info: Dict[str, Any], image_analysis: Dict[str, Any]) -> str:
        query_parts: List[str] = []
        if geo_info.get("region_name"):
            query_parts.append(str(geo_info["region_name"]))
        if image_analysis.get("parameter_type"):
            query_parts.append(str(image_analysis["parameter_type"]))
        for anomaly in image_analysis.get("anomalies", []):
            if isinstance(anomaly, dict):
                if anomaly.get("anomaly_type"):
                    query_parts.append(str(anomaly["anomaly_type"]))
                if anomaly.get("description"):
                    query_parts.append(str(anomaly["description"]))
        return " ".join([p for p in query_parts if p]).strip()

    def _generate_initial_interpretation(self, geo_info: Dict[str, Any], image_analysis: Dict[str, Any]) -> str:
        prompt = (
            "你是一位专业地球物理学家。请基于用户反演结果的结构化JSON，给出“初步/简单解释”。\n"
            "要求：\n"
            "1) 只依据输入JSON与地理范围，不要编造文献或外部事实\n"
            "2) 每条解释都要明确对应到哪些异常特征（用异常序号或异常类型）\n"
            "3) 使用谨慎措辞，显式标注不确定性\n"
            "4) 输出为Markdown，包含：关键异常概述、可能地质含义(2-5条)、需要补充的数据(3条)\n"
            "\n"
            f"研究区域：{geo_info.get('region_name', '未指定')}\n"
            f"经度范围：{geo_info.get('longitude_range', [])}\n"
            f"纬度范围：{geo_info.get('latitude_range', [])}\n"
            f"深度范围：{geo_info.get('depth_range', [])}\n"
            "\n"
            "反演结果JSON：\n"
            "```json\n"
            f"{json.dumps(image_analysis, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
        response = self.text_model.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()

    def _generate_hypotheses(
        self,
        geo_info: Dict[str, Any],
        image_analysis: Dict[str, Any],
        initial_interpretation: str,
        branches: int = 3,
    ) -> List[Dict[str, Any]]:
        prompt = (
            "你是一位资深地球物理学家，正在用思维树(Tree-of-Thought)为反演图做多分支解释。\n"
            "请生成多个相互竞争的解释假设，并用结构化JSON输出。\n"
            "约束：\n"
            "1) 不要引用或虚构任何文献\n"
            "2) 每个假设必须明确：机制、与观测匹配点、冲突点/薄弱处、可检验预测、需要的新增数据\n"
            "3) 假设之间要有差异（例如：部分熔融/流体富集/板片几何变化/温度异常/构造弱带等）\n"
            f"4) 生成 {branches} 个假设\n"
            "\n"
            "输出JSON格式（严格遵守，不要额外文本）：\n"
            "[\n"
            "  {\n"
            '    "id": "H1",\n'
            '    "title": "string",\n'
            '    "mechanism": "string",\n'
            '    "supported_by": ["string"],\n'
            '    "conflicts_or_gaps": ["string"],\n'
            '    "testable_predictions": ["string"],\n'
            '    "required_additional_data": ["string"],\n'
            '    "search_keywords": ["string"]\n'
            "  }\n"
            "]\n"
            "\n"
            f"研究区域：{geo_info.get('region_name', '未指定')}\n"
            "\n"
            "反演结果JSON：\n"
            "```json\n"
            f"{json.dumps(image_analysis, ensure_ascii=False, indent=2)}\n"
            "```\n"
            "\n"
            "初步解释（供参考，可被推翻）：\n"
            f"{initial_interpretation}\n"
        )
        model = self.text_model.bind(temperature=0.7)
        response = model.invoke([HumanMessage(content=prompt)])
        data = self._extract_json_from_text(str(response.content))
        if not isinstance(data, list):
            raise ValueError("假设生成输出不是JSON数组")
        return data

    def _evaluate_hypotheses(
        self,
        geo_info: Dict[str, Any],
        image_analysis: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = (
            "你是一位严谨的地球物理审稿人。请对多个解释假设进行评估与排序。\n"
            "约束：\n"
            "1) 只依据反演结果JSON与假设内容，不要引入外部事实或文献\n"
            "2) 给出可解释性与可检验性导向的评分\n"
            "\n"
            "评分维度(0-10)：\n"
            "- consistency: 与观测异常匹配程度\n"
            "- parsimony: 假设简约性/额外假设最少\n"
            "- plausibility: 地球物理合理性\n"
            "- falsifiability: 可证伪/可检验程度\n"
            "- risk: 关键不确定性与失败风险(分数越高风险越大)\n"
            "\n"
            "输出JSON格式（严格遵守，不要额外文本）：\n"
            "{\n"
            '  "ranking": [{"id": "H1", "total_score": 0, "rationale": "string"}],\n'
            '  "score_breakdown": [{"id": "H1", "consistency": 0, "parsimony": 0, "plausibility": 0, "falsifiability": 0, "risk": 0}],\n'
            '  "key_missing_info": ["string"],\n'
            '  "recommended_next_steps": ["string"]\n'
            "}\n"
            "\n"
            f"研究区域：{geo_info.get('region_name', '未指定')}\n"
            "\n"
            "反演结果JSON：\n"
            "```json\n"
            f"{json.dumps(image_analysis, ensure_ascii=False, indent=2)}\n"
            "```\n"
            "\n"
            "候选假设JSON：\n"
            "```json\n"
            f"{json.dumps(hypotheses, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
        model = self.text_model.bind(temperature=0.2)
        response = model.invoke([HumanMessage(content=prompt)])
        data = self._extract_json_from_text(str(response.content))
        if not isinstance(data, dict):
            raise ValueError("评估输出不是JSON对象")
        return data

    def _generate_report_workflow_i(
        self,
        geo_info: Dict[str, Any],
        image_analysis: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        evaluation: Dict[str, Any],
        initial_interpretation: str,
    ) -> str:
        template = self._read_prompt_template("report_generation")
        background = geo_info.get("geological_background") or geo_info.get("background") or "未提供"
        prompt = (
            f"{template}\n\n"
            "## 具体输入数据\n"
            "### 用户反演结果（结构化JSON）\n"
            "```json\n"
            f"{json.dumps(image_analysis, ensure_ascii=False, indent=2)}\n"
            "```\n"
            "\n"
            "### 相关文献\n"
            "未提供（本次为Workflow I，仅做看图初步解释）\n"
            "\n"
            "### 区域地质背景\n"
            f"{background}\n"
            "\n"
            "### 思维树候选假设（JSON）\n"
            "```json\n"
            f"{json.dumps(hypotheses, ensure_ascii=False, indent=2)}\n"
            "```\n"
            "\n"
            "### 假设评估与排序（JSON）\n"
            "```json\n"
            f"{json.dumps(evaluation, ensure_ascii=False, indent=2)}\n"
            "```\n"
            "\n"
            "### 初步解释（Markdown）\n"
            f"{initial_interpretation}\n"
        )
        response = self.text_model.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()

def main():
    """主函数"""
    base_dir = Path(__file__).resolve().parent
    default_image = str((base_dir / "examples" / "image.png").resolve())
    default_geo_info = str((base_dir / "examples" / "test_geo_info.json").resolve())
    default_output = str((base_dir / "examples" / "out_report.md").resolve())

    parser = argparse.ArgumentParser(description="地球物理反演解释Agent")
    parser.add_argument("--image", default=default_image, help="反演结果图像路径")
    parser.add_argument("--geo-info", default=default_geo_info, help="地理信息JSON文件路径")
    parser.add_argument("--output", default=default_output, help="输出报告路径")
    parser.add_argument("--branches", type=int, default=3, help="思维树分支数")
    
    args = parser.parse_args()
    
    # 加载地理信息
    with open(args.geo_info, 'r', encoding='utf-8') as f:
        geo_info = json.load(f)
    
    # 验证必需字段
    required_fields = ['longitude_range', 'latitude_range', 'depth_range', 'region_name']
    for field in required_fields:
        if field not in geo_info:
            raise ValueError(f"地理信息缺少必需字段: {field}")
    
    # 创建Agent并处理
    try:
        agent = EarthPhysicsAgent(enable_literature=False)
        result = agent.run_workflow_i(args.image, geo_info, branches=args.branches)
    except Exception as e:
        if openai is not None and isinstance(e, getattr(openai, "AuthenticationError", ())):
            print("阿里云模型鉴权失败：请检查 config.py/.env 或环境变量中的 ALIBABA_API_KEY / DASHSCOPE_API_KEY。")
            sys.exit(2)
        if isinstance(e, ValueError) and "API Key" in str(e):
            print(str(e))
            sys.exit(2)
        raise
    
    # 保存报告
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(result["final_report"])
    
    print(f"\n分析完成！报告已保存到: {args.output}")

if __name__ == "__main__":
    main()
