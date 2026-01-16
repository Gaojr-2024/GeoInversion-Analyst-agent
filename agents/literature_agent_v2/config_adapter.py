import sys
import os
from pathlib import Path

# 将项目根目录添加到 sys.path
current_file = Path(__file__).resolve()
literature_agent_v2_dir = current_file.parent
agents_dir = literature_agent_v2_dir.parent
project_root = agents_dir.parent # langchain_version

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import (
        ALIBABA_API_KEY,
        OPENAI_COMPATIBLE_BASE_URL,
        TEXT_MODEL,
        EMBEDDING_MODEL
    )
except ImportError:
    # 尝试从上级目录导入 (如果项目结构不同)
    sys.path.append(str(project_root.parent))
    from langchain_version.config import (
        ALIBABA_API_KEY,
        OPENAI_COMPATIBLE_BASE_URL,
        TEXT_MODEL,
        EMBEDDING_MODEL
    )

# 确保 API Key 存在
if not ALIBABA_API_KEY:
    raise ValueError("未找到 ALIBABA_API_KEY，请检查 langchain_version/config.py 或环境变量")
