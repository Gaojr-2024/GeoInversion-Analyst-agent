"""
多模态图像分析器 - 使用LangChain集成视觉大模型
支持两种使用场景：
1. 用户输入分析：分析用户提供的反演结果图像
2. 文献图像分析：分析从文献库中提取的图像
"""

import os
import json
import base64
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 动态导入config模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VISION_MODEL, ALIBABA_API_KEY, OPENAI_COMPATIBLE_BASE_URL, PROMPT_TEMPLATES

class MultiModalAnalyzer:
    """多模态图像分析器，用于分析地球物理反演结果图和文献图像"""
    
    def __init__(self):
        """初始化视觉模型"""
        if not ALIBABA_API_KEY:
            raise ValueError("缺少阿里云 API Key：请在 config.py/.env/环境变量中配置")
            
        # 使用阿里云的OpenAI兼容API
        self.vision_model = ChatOpenAI(
            model=VISION_MODEL,
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            temperature=0.1,
            max_tokens=10000
        )
        
        print(f"使用阿里云OpenAI兼容API: {VISION_MODEL}")
    
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
    
    def _build_vision_prompt(self, geo_info: Optional[Dict[str, Any]] = None) -> str:
        """
        构建视觉分析提示词
        
        Args:
            geo_info: 地理信息字典（可选）
            
        Returns:
            完整的提示词
        """
        # 读取提示词模板
        prompt_template_path = PROMPT_TEMPLATES["vision_analysis"]
        if os.path.exists(prompt_template_path):
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                template = f.read()
        else:
            raise FileNotFoundError(f"提示词模板文件不存在: {prompt_template_path}")
        
        if geo_info:
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
            
            # 替换模板中的所有占位符
            prompt = template.replace("{user_geo_info_context}", user_geo_context)
            prompt = prompt.replace("{user_longitude_min}", str(user_longitude_min))
            prompt = prompt.replace("{user_longitude_max}", str(user_longitude_max))
            prompt = prompt.replace("{user_latitude_min}", str(user_latitude_min))
            prompt = prompt.replace("{user_latitude_max}", str(user_latitude_max))
            prompt = prompt.replace("{user_depth_min}", str(user_depth_min))
            prompt = prompt.replace("{user_depth_max}", str(user_depth_max))
        else:
            # 没有地理信息时，使用通用提示词
            prompt = template.replace("{user_geo_info_context}", "- 研究区域: 未指定\n- 经度范围: N/A\n- 纬度范围: N/A\n- 深度范围: N/A")
            prompt = prompt.replace("{user_longitude_min}", "N/A")
            prompt = prompt.replace("{user_longitude_max}", "N/A")
            prompt = prompt.replace("{user_latitude_min}", "N/A")
            prompt = prompt.replace("{user_latitude_max}", "N/A")
            prompt = prompt.replace("{user_depth_min}", "N/A")
            prompt = prompt.replace("{user_depth_max}", "N/A")
        
        return prompt
    
    def analyze_inversion_image(
        self, 
        image_path: str, 
        user_geo_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        分析地球物理反演结果图像（用户输入场景）
        
        Args:
            image_path: 图像文件路径
            user_geo_info: 用户提供的地理信息（可选）
            
        Returns:
            分析结果字典
        """
        # 验证图像
        self.validate_image(image_path)
        
        # 编码图像
        image_base64 = self.encode_image_to_base64(image_path)
        
        # 构建提示词
        prompt = self._build_vision_prompt(user_geo_info)
        
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
            response = self.vision_model.invoke(messages)
            response_content = response.content
            result = self.extract_json_from_response(response_content)
            return result
        except Exception as e:
            print(f"视觉分析失败: {e}")
            raise
    
    def analyze_literature_image(
        self,
        image_path: str,
        literature_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析从文献中提取的图像（文献检索场景）
        
        Args:
            image_path: 文献图像文件路径
            literature_context: 文献上下文信息（可选）
            
        Returns:
            分析结果字典
        """
        # 验证图像
        self.validate_image(image_path)
        
        # 编码图像
        image_base64 = self.encode_image_to_base64(image_path)
        
        # 构建提示词（无地理信息约束）
        prompt = self._build_vision_prompt(None)
        
        # 如果有文献上下文，添加到提示词中
        if literature_context:
            prompt += f"\n\n## 文献上下文\n{literature_context}"
        
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
            response = self.vision_model.invoke(messages)
            response_content = response.content
            result = self.extract_json_from_response(response_content)
            return result
        except Exception as e:
            print(f"文献图像分析失败: {e}")
            raise

# 使用示例
if __name__ == "__main__":
    analyzer = MultiModalAnalyzer()
    
    # 测试用户输入场景
    test_geo_info = {
        "longitude_range": [116.0, 118.0],
        "latitude_range": [39.0, 41.0],
        "depth_range": [0, 50],
        "region_name": "华北克拉通东部"
    }
    
    # 假设有一个测试图像
    test_image_path = "../examples/image.png"
    if os.path.exists(test_image_path):
        result = analyzer.analyze_inversion_image(test_image_path, test_geo_info)
        print("用户输入分析结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 测试文献图像场景
    literature_result = analyzer.analyze_literature_image(test_image_path, "这是一篇关于华北克拉通的研究论文")
    print("\n文献图像分析结果:")
    print(json.dumps(literature_result, indent=2, ensure_ascii=False))
