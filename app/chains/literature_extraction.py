import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, TEXT_MODEL, get_prompt_path

class LiteratureExtractionChain:
    """
    从全文提取关键信息和图表证据
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            model_name=TEXT_MODEL,
            temperature=0.1,
            max_tokens=4000
        )
        self.prompt_template = self._load_prompt("literature_extraction")
        self.prompt = PromptTemplate(
            input_variables=["full_text"],
            template=self.prompt_template + "\n\n【全文内容】\n{full_text}"
        )

    def _load_prompt(self, prompt_name: str) -> str:
        path = get_prompt_path(prompt_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {path}")

    def extract_info(self, full_text: str) -> Dict[str, Any]:
        """
        执行提取
        """
        # Truncate full_text if too long to avoid token limits (naive truncation)
        # Assuming 1 token ~= 4 chars, 32k tokens ~= 128k chars. 
        # Safety limit: 100k chars.
        if len(full_text) > 100000:
            full_text = full_text[:100000] + "...(truncated)"

        chain = self.prompt | self.llm | StrOutputParser()
        
        try:
            res = chain.invoke({"full_text": full_text})
            
            # Clean JSON
            res = res.strip()
            if res.startswith("```json"):
                res = res[7:]
            if res.endswith("```"):
                res = res[:-3]
            
            return json.loads(res.strip())
        except Exception as e:
            print(f"Literature Extraction Failed: {e}")
            return {}
