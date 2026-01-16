import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from .config_adapter import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, TEXT_MODEL

class QueryGenerator:
    def __init__(self, prompt_path: Optional[str] = None):
        self.llm = ChatOpenAI(
            model=TEXT_MODEL,
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            temperature=0.0 # 强制 0.0 以保证输出一致性
        )
        # 动态获取 prompt 路径
        if prompt_path:
            self.prompt_path = Path(prompt_path)
        else:
            # 动态获取 prompt 路径 (langchain_version/prompts)
            # __file__ -> literature_agent_v2 -> agents -> langchain_version
            self.prompt_path = Path(__file__).resolve().parent.parent.parent / "prompts" / "literature_v2_query_generator.txt"

    def generate_queries(self, wf1_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """根据 Workflow I 的结果生成检索策略包 (包含 queries 和 negative_terms)"""
        
        # 读取 Prompt 模板
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt 文件未找到: {self.prompt_path}")
        
        template = self.prompt_path.read_text(encoding="utf-8")
        
        # 准备数据
        geo_info = wf1_bundle.get("geo_info", {})
        image_analysis = wf1_bundle.get("image_analysis", {})
        wf1_explain = wf1_bundle.get("initial_explain", "")
        
        # 构造 image_info_json
        image_info = {
            "region_names": geo_info.get("region_name", ""),
            "lonlat_bounds": geo_info.get("bbox", ""),
            "anomalies": image_analysis.get("anomalies", [])
        }
        
        # 填充模板
        prompt_content = template.replace("{{IMAGE_INFO_JSON}}", json.dumps(image_info, ensure_ascii=False))
        prompt_content = prompt_content.replace("{{WF1_SIMPLE_EXPLAIN}}", wf1_explain)
        prompt_content = prompt_content.replace("{{USER_KEYWORDS}}", geo_info.get("user_keywords", ""))
        
        # 兼容旧模板变量 (以防万一)
        prompt_content = prompt_content.replace("{{wf1_bundle_json}}", json.dumps(wf1_bundle, ensure_ascii=False))
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt_content)])
            content = response.content.strip()
            
            # 提取 JSON
            content = content.replace("```json", "").replace("```", "").strip()
            
            # 尝试提取字典 {}
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(content[start:end])
            
            # 尝试提取列表 [] (兼容旧格式)
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != -1:
                queries = json.loads(content[start:end])
                if isinstance(queries, list):
                    return {"queries": queries, "negative_terms": []}
            
            print("  [Warning] 无法解析 Query Generator 的输出 (JSON not found)，使用基于地理位置的默认值。")
            print(f"  [Debug] LLM Output: {content[:200]}...") # 打印部分输出以便调试
            region = geo_info.get("region_name", "Unknown Region")
            return {"queries": [f"{region} geology", f"{region} seismic structure"], "negative_terms": []}
                
        except Exception as e:
            print(f"  [Error] 生成查询失败: {e}")
            region = geo_info.get("region_name", "Unknown Region")
            return {"queries": [f"{region} geology", f"{region} seismic structure"], "negative_terms": []}
