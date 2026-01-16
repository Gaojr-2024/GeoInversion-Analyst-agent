"""
用户输入处理器 - 处理文字和图片输入
"""

import os
import json
import base64
import re
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

try:
    import openai
except Exception:
    openai = None

# 动态导入config模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VISION_MODEL, ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL

class InputProcessor:
    """用户输入处理器，负责解析文字和图片输入"""
    
    def __init__(self):
        """初始化输入处理器"""
        if not ALIBABA_API_KEY:
            raise ValueError("缺少阿里云 API Key：请在 config.py/.env/环境变量中配置")
            
        self.vision_model = ChatOpenAI(
            model=VISION_MODEL,
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            temperature=0.3,
            max_tokens=10000
        )
        
        print(f"使用阿里云OpenAI兼容API: {VISION_MODEL}")
    
    def load_geo_info(self, geo_info_path: str) -> Dict[str, Any]:
        """
        加载用户提供的地理信息
        
        Args:
            geo_info_path: 地理信息JSON文件路径
            
        Returns:
            地理信息字典
        """
        if not os.path.exists(geo_info_path):
            raise FileNotFoundError(f"地理信息文件不存在: {geo_info_path}")
            
        with open(geo_info_path, 'r', encoding='utf-8') as f:
            geo_info = json.load(f)
            
        # 验证必需字段
        required_fields = ['longitude_range', 'latitude_range', 'depth_range', 'region_name']
        for field in required_fields:
            if field not in geo_info:
                raise ValueError(f"地理信息缺少必需字段: {field}")
                
        return geo_info
    
    def validate_image(self, image_path: str) -> bool:
        """
        验证图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            是否有效
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        ext = Path(image_path).suffix.lower()
        if ext not in valid_extensions:
            raise ValueError(f"不支持的图像格式: {ext}. 支持格式: {valid_extensions}")
            
        return True
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将图像编码为base64字符串
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            base64编码的字符串
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        从模型响应中提取JSON内容
        """
        # 方法1: 直接尝试解析
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # 方法2: 移除代码块标记后解析
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:].strip()
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:].strip()
            
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
            
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # 方法3: 查找第一个{到最后一个}
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        raise ValueError("无法从响应中提取有效的JSON")
    
    def analyze_inversion_image_with_vision_model(
        self,
        image_path: str,
        geo_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用视觉大模型分析反演结果图像
        
        Args:
            image_path: 图像文件路径
            geo_info: 地理信息字典
            
        Returns:
            分析结果字典
        """
        # 验证图像
        self.validate_image(image_path)
        
        # 编码图像
        image_base64 = self.encode_image_to_base64(image_path)
        
        # 构建提示词
        prompt = self._build_vision_prompt(geo_info)
        
        # 创建消息
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            )
        ]
        
        try:
            # 调用视觉模型
            response = self.vision_model.invoke(messages)
            response_content = response.content
            
            # 提取并解析JSON
            result = self.extract_json_from_response(response_content)
            return result
            
        except Exception as e:
            if openai is not None and isinstance(e, getattr(openai, "AuthenticationError", ())):
                raise ValueError(
                    "阿里云模型鉴权失败：请检查 config.py/.env 或环境变量中的 ALIBABA_API_KEY / DASHSCOPE_API_KEY。"
                ) from e
            print(f"视觉分析失败: {e}")
            raise
    
    def _build_vision_prompt(self, geo_info: Dict[str, Any]) -> str:
        """
        构建视觉分析提示词
        """
        # 提取地理信息范围
        longitude_range = geo_info.get('longitude_range', [])
        latitude_range = geo_info.get('latitude_range', [])
        depth_range = geo_info.get('depth_range', [])
        
        user_longitude_min = longitude_range[0] if longitude_range else 'N/A'
        user_longitude_max = longitude_range[1] if longitude_range else 'N/A'
        user_latitude_min = latitude_range[0] if latitude_range else 'N/A'
        user_latitude_max = latitude_range[1] if latitude_range else 'N/A'
        user_depth_min = depth_range[0] if depth_range else 'N/A'
        user_depth_max = depth_range[1] if depth_range else 'N/A'
        
        # 构建用户地理信息上下文
        user_geo_context = f"""- 研究区域: {geo_info.get('region_name', '未指定')}
- 经度范围: {user_longitude_min}°E - {user_longitude_max}°E
- 纬度范围: {user_latitude_min}°N - {user_latitude_max}°N
- 深度范围: {user_depth_min} - {user_depth_max} km"""
        
        base_dir = Path(__file__).resolve().parent
        prompt_template_path = (base_dir / ".." / "prompts" / "vision_analysis_prompt.txt").resolve()
        if os.path.exists(prompt_template_path):
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                template = f.read()
        else:
            raise FileNotFoundError(f"提示词模板文件不存在: {prompt_template_path}")
        
        extra_instruction = "\n请在生成的 JSON 中，特别是 description 字段中，给出更详细的地球物理解释，用 2-3 句话说明每个异常的含义、可能的地质成因以及与研究区域的关系。"
        template = template + extra_instruction

        prompt = template.replace("{user_geo_info_context}", user_geo_context)
        prompt = prompt.replace("{user_longitude_min}", str(user_longitude_min))
        prompt = prompt.replace("{user_longitude_max}", str(user_longitude_max))
        prompt = prompt.replace("{user_latitude_min}", str(user_latitude_min))
        prompt = prompt.replace("{user_latitude_max}", str(user_latitude_max))
        prompt = prompt.replace("{user_depth_min}", str(user_depth_min))
        prompt = prompt.replace("{user_depth_max}", str(user_depth_max))
        
        return prompt
    
def main():
    base_dir = Path(__file__).resolve().parent
    examples_dir = (base_dir / ".." / "examples").resolve()
    image_path = str(examples_dir / "image.png")
    geo_info_path = str(examples_dir / "test_geo_info.json")
    output_path = examples_dir / "image_analysis.json"

    processor = InputProcessor()
    geo_info = processor.load_geo_info(geo_info_path)
    result = processor.analyze_inversion_image_with_vision_model(image_path, geo_info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"分析完成，JSON结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
