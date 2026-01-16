import fitz  # PyMuPDF
import os
import json
import re
from pathlib import Path
from PIL import Image
import shutil

class PDFFigureExtractor:
    """
    基于图注锚定法的PDF图片提取器 (Tool)
    完全替代MinerU，实现精准抠图（含Colorbar和图例）
    """
    
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True, parents=True)
        
    def process_pdf(self, pdf_path: str) -> Path:
        """处理单个PDF文件"""
        pdf_path = Path(pdf_path)
        paper_id = pdf_path.stem
        if paper_id.endswith("_origin"):
            paper_id = paper_id[:-7]
            
        output_dir = self.output_base_dir / paper_id
        images_dir = output_dir / "images"
        
        # Clean up old results
        if output_dir.exists():
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(f"Warning: Failed to clean output dir {output_dir}: {e}")
        
        images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"正在处理: {pdf_path.name}")
        
        doc = fitz.open(pdf_path)
        extracted_figures = []
        full_text_blocks = [] 
        
        for page_idx, page in enumerate(doc):
            # 1. 获取页面所有文本块
            text_blocks = page.get_text("blocks") # (x0, y0, x1, y1, text, block_no, block_type)
            
            # 2. 预加载页面的所有视觉元素
            drawings = [path["rect"] for path in page.get_drawings()]
            images_info = page.get_image_info()
            image_rects = [fitz.Rect(img["bbox"]) for img in images_info]
            visual_elements = drawings + image_rects
            
            # 3. 识别图注 (Figure Caption)
            captions = []
            for block in text_blocks:
                text = block[4].strip()
                # Match Figure X, Fig. X, FIG. X, etc.
                if re.match(r'^(Figure|Fig\.|FIG\.)\s*\d+', text, re.IGNORECASE):
                    captions.append(block)
            
            page_figures = []
            
            for cap_block in captions:
                cap_rect = fitz.Rect(cap_block[:4])
                cap_text = cap_block[4].strip()
                
                # 定义搜索区域
                y_bottom = cap_rect.y0
                
                # 确定搜索的左右边界（处理分栏）
                page_width = page.rect.width
                
                is_full_width = (cap_rect.width > page_width * 0.6)
                margin = 30 
                
                if is_full_width:
                    search_x0 = 0
                    search_x1 = page_width
                else:
                    if cap_rect.x0 < page_width / 2: # Left column
                        search_x0 = 0
                        search_x1 = page_width / 2 + margin 
                    else: # Right column
                        search_x0 = page_width / 2 - margin
                        search_x1 = page_width
                
                # 寻找 y_top (Smart Logic)
                # 目标：找到上方最近的"正文"文本，忽略图内文字(Labels)
                closest_text_y1 = 0
                
                for tb in text_blocks:
                    tb_rect = fitz.Rect(tb[:4])
                    if tb_rect == cap_rect:
                        continue
                        
                    # 必须在图注上方
                    if tb_rect.y1 <= y_bottom:
                        # 必须在水平范围内
                        if not (tb_rect.x1 < search_x0 or tb_rect.x0 > search_x1):
                            
                            # 过滤逻辑：判断是否为图内文字
                            # 1. 如果文本块与任何视觉元素有显著重叠，视为图内文字
                            is_label = False
                            for vis in visual_elements:
                                if tb_rect.intersects(vis):
                                    # 如果重叠面积较大，或者是包含关系
                                    intersection = tb_rect & vis
                                    if intersection.get_area() > tb_rect.get_area() * 0.5:
                                        is_label = True
                                        break
                            
                            # 2. 如果文本非常短且靠近某个视觉元素（比如距离<10px），视为Label
                            if not is_label and len(tb[4].strip()) < 50:
                                # 检查距离
                                for vis in visual_elements:
                                    # 简单距离检查
                                    dist_x = max(0, tb_rect.x0 - vis.x1, vis.x0 - tb_rect.x1)
                                    dist_y = max(0, tb_rect.y0 - vis.y1, vis.y0 - tb_rect.y1)
                                    if dist_x < 10 and dist_y < 10:
                                        is_label = True
                                        break
                                        
                            if not is_label:
                                if tb_rect.y1 > closest_text_y1:
                                    closest_text_y1 = tb_rect.y1
                
                y_top = closest_text_y1
                
                # 安全检查：如果y_top太靠近y_bottom，可能误判了，强制留出空间
                if y_bottom - y_top < 100:
                    # 尝试向上多看一点，或者忽略y_top限制，使用固定高度搜索
                    # 比如向上搜索 400px (半页高)
                    y_top = max(0, y_bottom - 500)
                
                search_rect = fitz.Rect(search_x0, y_top, search_x1, y_bottom)
                
                # 4. 空间聚类：基于种子生长算法 (Seed Growth / Flood Fill)
                union_rect = fitz.Rect()
                has_elements = False
                
                # 步骤 1: 识别种子元素 (Seeds)
                seed_elements = []
                candidate_elements = [] 
                
                for element_rect in visual_elements:
                    # 排除全页背景
                    if element_rect.get_area() > (page.rect.get_area() * 0.9):
                        continue
                    
                    # 只有位于图注上方(或略微重叠)的元素才考虑
                    if element_rect.y0 > y_bottom + 20: 
                        continue
                        
                    # 只有位于上一段文本下方(或略微重叠)的元素才考虑
                    if element_rect.y1 < y_top - 20:
                        continue
                        
                    intersection = element_rect & search_rect
                    overlap_area = intersection.get_area()
                    element_area = element_rect.get_area()
                    
                    if element_area <= 0:
                        continue
                        
                    # 种子判据
                    is_mostly_inside = (overlap_area / element_area) > 0.2
                    is_inside = search_rect.contains(element_rect)
                    center_inside = search_rect.contains(fitz.Point((element_rect.x0+element_rect.x1)/2, (element_rect.y0+element_rect.y1)/2))
                    significant_overlap = overlap_area > 300 
                    
                    if intersection.get_area() > 0:
                        if is_mostly_inside or is_inside or center_inside or significant_overlap:
                            seed_elements.append(element_rect)
                        else:
                            candidate_elements.append(element_rect)
                    else:
                        # 即使不相交，如果垂直位置合适，也作为候选
                        # 放宽水平限制，因为Colorbar可能在Side
                        if abs(element_rect.x1 - search_rect.x0) < page_width/2 or abs(element_rect.x0 - search_rect.x1) < page_width/2:
                            # 垂直方向必须在 search_rect 范围内（或者稍微延伸）
                            if element_rect.y1 > y_top and element_rect.y0 < y_bottom:
                                candidate_elements.append(element_rect)

                # Fallback seeds
                if not seed_elements:
                    expanded_rect = fitz.Rect(search_rect)
                    expanded_rect.y0 = max(0, expanded_rect.y0 - 150)
                    for element_rect in visual_elements:
                        if element_rect.get_area() > (page.rect.get_area() * 0.9): continue
                        if element_rect.intersects(expanded_rect):
                            seed_elements.append(element_rect)
                
                # 步骤 2: 迭代生长 (Clustering)
                if seed_elements:
                    current_cluster = list(seed_elements)
                    cluster_rect = fitz.Rect()
                    for r in current_cluster:
                        if cluster_rect.is_empty:
                            cluster_rect = r
                        else:
                            cluster_rect |= r
                    
                    changed = True
                    while changed:
                        changed = False
                        next_candidates = []
                        
                        # 吸附区域：大幅增加垂直扩展，以跨越Gap
                        # 增加 expansion_x 以处理较远的 Colorbar
                        # 增加 expansion_y 以处理被 Label 分隔的图块
                        expansion_x = 80 
                        expansion_y = 80 
                        search_zone = cluster_rect + (-expansion_x, -expansion_y, expansion_x, expansion_y)
                        
                        for cand in candidate_elements:
                            # 只要相交或者被包含在 search_zone 内
                            if cand.intersects(search_zone):
                                current_cluster.append(cand)
                                cluster_rect |= cand
                                changed = True
                            else:
                                next_candidates.append(cand)
                        
                        candidate_elements = next_candidates
                    
                    union_rect = cluster_rect
                    has_elements = True
                
                if has_elements:
                    # 增加 Padding 防止边缘裁剪
                    padding = 15
                    final_crop_box = union_rect + (-padding, -padding, padding, padding)
                    final_crop_box = final_crop_box & page.rect
                    
                    # 高清截图
                    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=final_crop_box)
                    
                    fig_num_match = re.search(r'\d+', cap_text)
                    fig_num = fig_num_match.group(0) if fig_num_match else f"p{page_idx}_unk"
                    img_filename = f"Figure_{fig_num}.jpg"
                    img_path = images_dir / img_filename
                    
                    pix.save(str(img_path))
                    
                    figure_info = {
                        "img_name": img_filename,
                        "caption": cap_text,
                        "page": page_idx + 1,
                        "bbox": [final_crop_box.x0, final_crop_box.y0, final_crop_box.x1, final_crop_box.y1],
                        "img_path": f"images/{img_filename}"
                    }
                    extracted_figures.append(figure_info)
                    page_figures.append(figure_info)
                    
            # Generate Markdown
            text_blocks.sort(key=lambda b: (b[1], b[0]))
            page_md = f"## Page {page_idx + 1}\n\n"
            for block in text_blocks:
                text = block[4].strip()
                is_caption = False
                matched_fig = None
                for fig in page_figures:
                    if fig["caption"] in text:
                         is_caption = True
                         matched_fig = fig
                         break
                
                if is_caption and matched_fig:
                    page_md += f"\n\n!['{matched_fig['caption']}']({matched_fig['img_path']})\n\n"
                    page_md += f"**{matched_fig['caption']}**\n\n"
                else:
                    page_md += text + "\n\n"
            
            full_text_blocks.append(page_md)

        # Save outputs
        with open(output_dir / "content_list.json", "w", encoding="utf-8") as f:
            json.dump(extracted_figures, f, indent=2, ensure_ascii=False)
            
        with open(output_dir / "full.md", "w", encoding="utf-8") as f:
            f.write(f"# {paper_id}\n\n")
            f.write("\n---\n".join(full_text_blocks))
            
        print(f"处理完成: {output_dir}")
        return output_dir
