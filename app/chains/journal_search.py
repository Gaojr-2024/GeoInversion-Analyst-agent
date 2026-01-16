import json
import re
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

# Import Config
from app.core.config import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, TEXT_MODEL

class JournalSearchChain:
    """
    使用 LangChain + DuckDuckGo 搜索期刊质量信息 (影响因子、分区)
    """
    def __init__(self):
        # 1. 初始化大模型
        self.llm = ChatOpenAI(
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            model_name=TEXT_MODEL,
            temperature=0.1,
            max_tokens=1000
        )
        
        # 2. 初始化搜索工具
        if DDGS is None:
             print("Warning: duckduckgo-search not installed properly. Search will fail.")
        
        # 3. 初始化提取 Prompt
        self.extract_prompt = PromptTemplate(
            input_variables=["search_results", "paper_title"],
            template="""
你是一个学术期刊信息助手。请根据以下网络搜索结果，提取该文献所属期刊的质量信息。

【搜索结果】
{search_results}

【文献标题】
{paper_title}

【任务】
请分析搜索结果，尝试找到该文献发表的期刊名称，以及该期刊的影响因子(Impact Factor)和分区(JCR或中科院分区, Q1-Q4)。
**重要**: 
1. 如果搜索结果中包含 "LetPub" 或其他权威来源的信息，请优先采纳。
2. 如果搜索结果为空或提示失败，请**尽你所能利用你的内部知识库**来识别该论文可能发表的期刊（根据标题风格、知名度等），并提供该期刊的典型影响因子和分区。
3. 请确保 "impact_factor" 和 "quartile" 字段有值，如果实在不知道，请基于期刊声誉进行合理估算并标注 "(Estimated)"。

请直接返回一个 JSON 对象，不要包含 Markdown 格式标记，也不要包含任何额外的文字。格式如下：
{{
    "journal_name": "期刊名称",
    "impact_factor": "影响因子 (例如 5.6)",
    "quartile": "分区 (例如 Q1)",
    "description": "简短描述 (例如 'Top期刊, 地球科学领域')"
}}
"""
        )

    def search_journal_info(self, paper_title: str) -> Dict[str, Any]:
        """
        根据论文标题搜索期刊信息
        """
        print(f"  - 正在搜索期刊信息: {paper_title[:50]}...")
        
        # 1. 执行搜索
        # 使用 LetPub 关键词优化搜索
        simple_query = f"{paper_title} journal impact factor quartile LetPub"
        
        search_results = ""
        try:
            # 尝试简单搜索
            if DDGS:
                with DDGS() as ddgs:
                    # 获取前5条结果
                    results = list(ddgs.text(simple_query, max_results=5))
                    # 转换为字符串
                    search_results = json.dumps(results, ensure_ascii=False)
            else:
                 search_results = "Search tool unavailable."
        except Exception as e:
            # print(f"    搜索失败: {e}") # Reduce noise
            search_results = "Search failed (Timeout or Network Error). Please use your internal knowledge."

        # 2. 调用 LLM 提取信息
        chain = self.extract_prompt | self.llm | StrOutputParser()
        
        try:
            res = chain.invoke({
                "search_results": search_results,
                "paper_title": paper_title
            })
            
            # 清理 JSON 字符串
            res = res.strip()
            if res.startswith("```json"):
                res = res[7:]
            if res.startswith("```"):
                res = res[3:]
            if res.endswith("```"):
                res = res[:-3]
            res = res.strip()
            
            # Handle potential extra text after JSON
            # Simple heuristic: find last '}'
            last_brace = res.rfind("}")
            if last_brace != -1:
                res = res[:last_brace+1]
            
            data = json.loads(res)
            return data
            
        except Exception as e:
            print(f"    解析期刊信息失败: {e}")
            return {
                "journal_name": "Unknown", 
                "impact_factor": "Unknown", 
                "quartile": "Unknown",
                "raw_search": search_results[:100]
            }
