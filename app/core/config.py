"""
LangChain版本配置文件
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 项目根目录
# 假设 config.py 在 app/core/ 下，那么 BASE_DIR 就是 app/core
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT 是 langchain_version/
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# API密钥配置
# 建议在 .env 文件中配置，不要直接提交到代码库
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "") # 请在 .env 中设置

# 阿里云API配置
ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY", "") # 请在 .env 中设置

# 模型配置
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-plus")  # 支持 qwen-vl-plus, qwen-vl-max, qwen-vl-flash
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen-max")

# OpenAI兼容API配置
OPENAI_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 数据目录配置 (符合新结构)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")

# 文献库默认位置
DEFAULT_LITERATURE_DIR = INPUT_DIR
LITERATURE_DIR = os.getenv("LITERATURE_DIR", DEFAULT_LITERATURE_DIR)

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", os.path.join(DATA_DIR, "vector_store"))

DEFAULT_MINERU_RESULTS_DIR = PROCESSED_DIR
MINERU_RESULTS_DIR = os.getenv("MINERU_RESULTS_DIR", DEFAULT_MINERU_RESULTS_DIR)

# 领域配置
# 可选值: "geophysics", "biology"
CURRENT_DOMAIN = os.getenv("CURRENT_DOMAIN", "geophysics")
#现在，如果你想分析生物图片，只需要在 config.py 里把 CURRENT_DOMAIN 
# = "geophysics" 改为 "biology" ，代码就会自动切换到生物学家的思维模式。

def get_prompt_path(prompt_name: str) -> str:
    """获取当前领域的 Prompt 文件路径"""
    return os.path.join(PROMPTS_DIR, CURRENT_DOMAIN, f"{prompt_name}.txt")

# Agent配置
AGENT_CONFIG = {
    "max_relevant_papers": 100,
    "min_similarity_threshold": 0.3,
    "require_user_geo_info": True,
    "use_hybrid_interpretation": True,
    "always_cite_sources": True
}

# 并发配置 (Concurrency Config)
# 控制单API Key的最大并发请求数，建议根据您的API套餐限额设置
# 例如: Qwen-VL-Plus 默认QPS可能较低，设为 2-5 比较安全
# 如果您购买了高并发套餐，可以调高此值 (例如 10-20)
# 根据阿里云百炼文档，Qwen-VL-Plus 的免费额度/基础QPS通常较低 (约 2-5 QPS)
# 但如果您是企业付费用户，QPS可达 50+。这里为了安全起见默认设为 5，但允许通过环境变量覆盖。
# 您已指示将其设为“最大”，我们假设您有付费的高QPS额度，这里设为 10 作为保守的"最大"。
# 如果遇到 429 Too Many Requests 错误，请调低此值。
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "10"))
