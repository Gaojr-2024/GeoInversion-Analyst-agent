"""
LangChain版本配置文件
"""

import os
from typing import List, Dict

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# API密钥配置
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "")

# 阿里云API配置
ALIBABA_API_KEY_FILE = ""
ALIBABA_API_KEY = ALIBABA_API_KEY_FILE or os.getenv("ALIBABA_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""


# 模型配置
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-plus")  # 支持 qwen-vl-plus, qwen-vl-max, qwen-vl-flash
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen-max")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")

# OpenAI兼容API配置
OPENAI_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 文献库配置
DEFAULT_LITERATURE_DIR = os.path.join(PROJECT_ROOT, "我的文献库", "01-文献")
LITERATURE_DIR = os.getenv("LITERATURE_DIR", DEFAULT_LITERATURE_DIR)

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", os.path.join(BASE_DIR, "vector_store"))

MINERU_RESULTS_DIR = os.getenv("MINERU_RESULTS_DIR", DEFAULT_MINERU_RESULTS_DIR)

# 地球物理参数配置
GEOPHYSICAL_PARAMETERS = {
    "velocity": ["velocity", "速度", "vp", "vs", "p-wave", "s-wave"],
    "conductivity": ["conductivity", "电导率", "resistivity", "电阻率"],
    "density": ["density", "密度"],
    "attenuation": ["attenuation", "衰减", "quality factor", "q值"],
    "anisotropy": ["anisotropy", "各向异性", "shear wave splitting"]
}

# Agent配置
AGENT_CONFIG = {
    "max_relevant_papers": 5,
    "min_similarity_threshold": 0.3,
    "require_user_geo_info": True,
    "use_hybrid_interpretation": True,
    "always_cite_sources": True
}

# 提示词模板路径
PROMPT_TEMPLATES = {
    "vision_analysis": os.path.join(BASE_DIR, "prompts", "vision_analysis_prompt.txt"),
    "literature_extraction": os.path.join(BASE_DIR, "prompts", "literature_extraction_prompt.txt"),
    "report_generation": os.path.join(BASE_DIR, "prompts", "report_generation_prompt.txt"),
    "query_planner": os.path.join(BASE_DIR, "prompts", "query_planner_prompt.txt"),
    "reranker": os.path.join(BASE_DIR, "prompts", "reranker_prompt.txt"),
}
