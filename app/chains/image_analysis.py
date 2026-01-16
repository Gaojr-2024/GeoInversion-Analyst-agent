import os
import base64
import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

# Import Config from core
from app.core.config import ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, VISION_MODEL, get_prompt_path

class ImageAnalysisChain:
    """
    使用 LangChain 的多模态能力分析文献图片。
    对应流程中的：分析图片 (Image Analysis)
    """
    def __init__(self):
        # 初始化多模态大模型
        # 注意: ChatOpenAI 实例通常是线程安全的，但在高并发下，
        # 最好在 analyze_image 中动态创建或确保其内部 session 处理得当。
        # 这里为了简单起见，我们重用同一个实例，因为 LangChain 的底层 httpx 客户端是线程安全的。
        self.llm = ChatOpenAI(
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            model_name=VISION_MODEL,
            temperature=0.1, # 分析图片需要精确
            max_tokens=2000
        )
        
        # 从外部文件加载 Prompt
        self.prompt_template_str = self._load_prompt("vision_analysis")

    def _load_prompt(self, prompt_name: str) -> str:
        """加载外部 Prompt 文件"""
        path = get_prompt_path(prompt_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Critical Error: Prompt file not found at {path}. Please check your configuration.")

    def _encode_image(self, image_path: str) -> str:
        """将图片编码为 Base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path: str, context: Dict[str, Any]) -> Dict:
        """
        使用 LangChain 调用 VLM 分析单张图片
        """
        try:
            # 1. 准备图片
            base64_image = self._encode_image(image_path)
            image_url = f"data:image/jpeg;base64,{base64_image}"
            
            # 2. 准备 Prompt
            prompt_text = self.prompt_template_str.format(
                caption=context.get("caption", ""),
                original_interpretation=context.get("original_interpretation", ""),
                paper_info=json.dumps(context.get("paper_info", {}), ensure_ascii=False)
            )

            # 3. 构建 LangChain 消息
            messages = [
                SystemMessage(content="你是一个能够精确读取科学图像的 AI 助手。请只输出 JSON。"),
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                )
            ]

            # 4. 调用模型
            response = self.llm.invoke(messages)
            content = response.content

            # 5. 解析 JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content)

        except Exception as e:
            print(f"Error analyzing image {image_path}: {e}")
            return {
                "is_inversion_image": False,
                "error": str(e)
            }
