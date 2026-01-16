import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from .config_adapter import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, TEXT_MODEL

class RelevanceScorer:
    def __init__(self, prompt_path: Optional[str] = None):
        self.llm = ChatOpenAI(
            model=TEXT_MODEL,
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            temperature=0.0 # 评分需要客观，低温度
        )
        if prompt_path:
            self.prompt_path = Path(prompt_path)
        else:
            # 动态获取 prompt 路径 (langchain_version/prompts)
            # __file__ -> literature_agent_v2 -> agents -> langchain_version
            self.prompt_path = Path(__file__).resolve().parent.parent.parent / "prompts" / "literature_v2_relevance_scorer.txt"

    def score_papers(self, wf1_bundle: Dict[str, Any], papers: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """对论文进行批量评分"""
        if not self.prompt_path.exists():
            # 尝试向上查找
            if Path("prompts/literature_v2_relevance_scorer.txt").exists():
                self.prompt_path = Path("prompts/literature_v2_relevance_scorer.txt")
            else:
                raise FileNotFoundError(f"Prompt 文件未找到: {self.prompt_path}")

        template = self.prompt_path.read_text(encoding="utf-8")
        scored_papers = []
        
        # 提取关键上下文，减少 Token 消耗
        context_str = json.dumps({
            "region": wf1_bundle.get("geo_info", {}).get("region_name"),
            "features": wf1_bundle.get("image_analysis", {}).get("anomalies", []),
            "interpretation": wf1_bundle.get("initial_explain")[:500] + "..." # 截断
        }, ensure_ascii=False)

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            print(f"  正在评分第 {i+1}-{min(i+batch_size, len(papers))} 篇 ...")
            
            # 构建仅包含必要信息的 batch 用于 prompt
            batch_for_prompt = [
                {"id": p["id"], "title": p["title"], "abstract": p["abstract"][:800]} # 摘要截断
                for p in batch
            ]
            
            prompt = template.replace("{{context_str}}", context_str)
            prompt = prompt.replace("{{papers_json}}", json.dumps(batch_for_prompt, ensure_ascii=False))
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                content = response.content.strip()
                
                # 提取 JSON
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        # 尝试修复常见的 JSON 错误 (如 Markdown 代码块残留, 中文引号等)
                        # 这里简单尝试清理 Markdown 标记
                        json_str = json_str.replace("```json", "").replace("```", "").strip()
                        # 修复潜在的 LLM 输出错误，如 backticks 在字符串内
                        # 示例: "id": "`https://...`" -> "id": "https://..."
                        json_str = json_str.replace("`", "") 
                        
                        # 再次尝试提取
                        start = json_str.find('{')
                        end = json_str.rfind('}') + 1
                        if start != -1 and end != -1:
                             json_str = json_str[start:end]
                             try:
                                result = json.loads(json_str)
                             except Exception as e_inner:
                                print(f"  [Error] Batch {i} JSON 解析失败 (二次尝试): {e_inner}")
                                print(f"  [Debug] Content snippet: {json_str[:200]}...") # Increased debug length
                                raise e_inner
                        else:
                             raise ValueError("JSON block not found after cleanup")

                    score_map = {str(item.get("id")).strip(): item for item in result.get("scores", [])}
                    
                    # 将分数合并回原论文对象
                    for p in batch:
                        # 确保 ID 匹配时忽略可能的格式差异
                        pid = str(p["id"]).strip()
                        score_info = score_map.get(pid, {"score": 0, "reason": "评分失败"})
                        p["score"] = score_info.get("score", 0)
                        p["reason"] = score_info.get("reason", "未提供理由")
                        scored_papers.append(p)
                else:
                    print(f"  [Warning] Batch {i} 评分格式错误")
                    for p in batch:
                        p["score"] = 0
                        p["reason"] = "评分格式解析失败"
                        scored_papers.append(p)
                        
            except Exception as e:
                print(f"  [Error] Batch {i} 评分异常: {e}")
                for p in batch:
                    p["score"] = 0
                    p["reason"] = f"评分过程异常: {str(e)}"
                    scored_papers.append(p)

        # 按分数降序排列
        scored_papers.sort(key=lambda x: x["score"], reverse=True)
        return scored_papers
