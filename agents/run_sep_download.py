import asyncio
import os
import sys
import re
import json
import time
import base64
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from playwright.async_api import async_playwright, Page, Response

# Import config (Need to add project root to path first)
current_file = Path(__file__).resolve()
current_dir = current_file.parent # agents
project_root = current_dir.parent # langchain_version

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from config import (
        ALIBABA_API_KEY, 
        VISION_MODEL, 
        OPENAI_COMPATIBLE_BASE_URL
    )
except ImportError:
    from ..config import (
        ALIBABA_API_KEY, 
        VISION_MODEL, 
        OPENAI_COMPATIBLE_BASE_URL
    )

from urllib.parse import urlparse

# Constants for UCAS WebVPN
# 使用 WebVPN 而不是 SEP 门户，因为只有 WebVPN 能提供校外文献访问权限
WEBVPN_LOGIN_URL = "https://webvpn.ucas.ac.cn/"

def convert_to_webvpn_url(original_url: str) -> str:
    """
    将普通 URL 转换为国科大 WebVPN 格式的 URL。
    规则通常为：
    1. 协议: https -> https
    2. 域名: . 替换为 -
    3. 如果是 https，域名后加 -s
    4. 后缀: .webvpn.ucas.ac.cn
    
    例如: 
    https://www.nature.com/articles/x -> https://www-nature-com-s.webvpn.ucas.ac.cn/articles/x
    http://example.com/path -> http://example-com.webvpn.ucas.ac.cn/path
    """
    if "webvpn.ucas.ac.cn" in original_url:
        return original_url
        
    try:
        parsed = urlparse(original_url)
        scheme = parsed.scheme
        netloc = parsed.netloc
        path = parsed.path
        query = parsed.query
        fragment = parsed.fragment
        
        # 处理域名: www.nature.com -> www-nature-com
        new_netloc = netloc.replace('.', '-')
        
        # 处理端口 (如果有)
        if ':' in new_netloc:
            new_netloc = new_netloc.replace(':', '-')
            
        # 处理 HTTPS 后缀
        if scheme == 'https':
            new_netloc += '-s'
            
        # 拼接 WebVPN 域名
        final_netloc = f"{new_netloc}.webvpn.ucas.ac.cn"
        
        # 重组 URL (WebVPN 通常统一使用 https 访问入口，但内部协议可能映射)
        # 这里保持原协议头通常更稳妥，或者统一用 https
        final_url = f"https://{final_netloc}{path}"
        
        if query:
            final_url += f"?{query}"
        if fragment:
            final_url += f"#{fragment}"
            
        return final_url
    except Exception as e:
        print(f"URL conversion failed: {e}, returning original.")
        return original_url

@dataclass
class PaperInfo:
    id: str # Changed to str to match download.py
    title: str
    score: int
    oa_status: str
    link: str
    year: str = ""
    author: str = ""
    journal: str = ""
    cited_by_count: int = 0
    abstract: str = ""
    download_status: str = "Pending"
    local_path: str = ""
    fail_reason: str = ""
    reason: str = "" # Scoring reason

class LiteratureParser:
    def __init__(self, md_path: str):
        self.md_path = md_path

    def parse(self) -> List[PaperInfo]:
        if not os.path.exists(self.md_path):
            print(f"File not found: {self.md_path}")
            return []
            
        with open(self.md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        papers = []
        current_paper = {}
        
        for line in lines:
            line = line.strip()
            # Match header: ### [Number]. [Title]
            # Updated pattern based on literature_review_with_downloads.md structure
            header_match = re.match(r'^###\s+(\d+)\.\s+(.+)', line)
            if header_match:
                if current_paper:
                    self._add_paper(papers, current_paper)
                
                current_paper = {
                    'id': str(header_match.group(1)),
                    'title': header_match.group(2).strip(),
                    'score': 0, 'year': "", 'author': "", 'journal': "", 'link': "", 'oa_status': "Unknown", 'reason': "", 'cited_by_count': 0
                }
                continue
            
            if not current_paper:
                continue

            # Extract Metadata
            if line.startswith('- **评分**:'):
                try:
                    current_paper['score'] = int(line.replace('- **评分**:', '').strip())
                except:
                    pass
            elif line.startswith('- **评分理由**:'):
                current_paper['reason'] = line.replace('- **评分理由**:', '').strip()
            elif line.startswith('- **来源**:'):
                content = line.replace('- **来源**:', '').strip()
                # Expected format from download.py: Journal Name (Year) Cited: 123
                # Regex to capture: Group 1 (Journal), Group 2 (Year), Group 3 (Count)
                # Note: Journal name might contain parentheses, so be careful.
                # Use strict matching for the end part first.
                
                # Match "Cited: 123" at the end
                cited_match = re.search(r'Cited:\s*(\d+)\s*$', content)
                if cited_match:
                    current_paper['cited_by_count'] = int(cited_match.group(1))
                    # Remove citation part
                    content = content[:cited_match.start()].strip()
                
                # Match "(Year)" at the end of remaining content
                year_match = re.search(r'\((\d{4})\)$', content)
                if year_match:
                    current_paper['year'] = year_match.group(1)
                    # The rest is the journal name
                    current_paper['journal'] = content[:year_match.start()].strip()
                else:
                    # Fallback logic if format differs
                    current_paper['journal'] = content

            elif line.startswith('- **链接**:'):
                link_match = re.search(r'\[(.*?)\]\((.*?)\)', line)
                if link_match:
                    current_paper['link'] = link_match.group(2)
            elif line.startswith('- **失败原因**:'):
                current_paper['fail_reason'] = line.replace('- **失败原因**:', '').strip()

        if current_paper:
            self._add_paper(papers, current_paper)
            
        return papers

    def _add_paper(self, papers, p):
        if 'title' in p and 'link' in p:
            papers.append(PaperInfo(
                id=p['id'],
                title=p['title'],
                score=p.get('score', 0),
                oa_status=p.get('oa_status', 'Unknown'),
                link=p.get('link', ''),
                year=p.get('year', ''),
                author=p.get('author', ''),
                journal=p.get('journal', ''),
                reason=p.get('reason', ''),
                cited_by_count=p.get('cited_by_count', 0)
            ))

    def parse_jsonl(self, jsonl_path: str) -> List[PaperInfo]:
        """Parses candidates_scored.jsonl generated by download.py"""
        if not os.path.exists(jsonl_path):
            return []
            
        papers = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Mapping logic mirroring download.py
                    authors = data.get("authors", [])
                    if isinstance(authors, list):
                        author_str = ", ".join(authors[:3])
                    else:
                        author_str = str(authors)

                    p = PaperInfo(
                        id=str(data.get("id", "")),
                        title=data.get("title", "Untitled"),
                        score=int(data.get("score", 0)),
                        oa_status="OA" if data.get("is_oa") else "Closed",
                        link=data.get("link") or data.get("pdf_url") or data.get("landing_page_url") or "",
                        year=str(data.get("year", "")),
                        author=author_str,
                        journal=data.get("venue", ""),
                        cited_by_count=data.get("cited_by_count", 0),
                        abstract=data.get("abstract", ""),
                        reason=data.get("reason", "")
                    )
                    papers.append(p)
                except Exception as e:
                    print(f"Error parsing JSONL line: {e}")
        return papers

class SEPDownloader:
    def __init__(self, download_dir: str):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        if not ALIBABA_API_KEY:
            raise ValueError("ALIBABA_API_KEY not set in config")
            
        self.vision_model = ChatOpenAI(
            model=VISION_MODEL,
            api_key=ALIBABA_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            temperature=0.1,
            max_tokens=2000
        )

    async def analyze_page(self, screenshot_b64: str) -> Dict[str, Any]:
        """
        Identify PDF download button OR PDF Preview state using VLM.
        """
        prompt = (
            "You are an automated web browsing assistant. Analyze this screenshot of an academic paper page.\n"
            "Task 1: Check for 'Verify you are human', 'Cloudflare', 'CAPTCHA', 'Access Denied', 'Please wait...'.\n"
            "Task 2: Check if this is a PDF PREVIEW interface. Look for:\n"
            "   - A top toolbar with icons like 'Print' (printer), 'Save' (floppy disk/arrow), 'Rotate', 'Page Number'.\n"
            "   - The main content looks like a document page (white paper on gray background).\n"
            "   - A small floating PDF icon in the bottom right corner.\n"
            "Task 3: Find the PDF download button. \n"
            "   - Look for distinct clickable BUTTONS (often rectangular, with black/blue/red background) labeled 'PDF' or 'Download PDF' or 'Full Text'.\n"
            "   - Also look for the 'Save'/'Download' icon in the top toolbar if it is a PDF preview.\n"
            "   - IGNORE text statistics like 'PDF Downloads (297)'.\n"
            "\n"
            "Return JSON:\n"
            "{\n"
            "  \"verification_present\": boolean,\n"
            "  \"is_pdf_preview\": boolean,\n"
            "  \"download_button_coordinates\": [ymin, xmin, ymax, xmax] (0-1000 scale) or null,\n"
            "  \"reason\": \"explanation\"\n"
            "}"
        )
        
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
        ])
        
        try:
            response = await self.vision_model.ainvoke([msg])
            content = response.content
            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            return json.loads(json_str)
        except Exception as e:
            print(f"Vision analysis failed: {e}")
            return {"verification_present": False, "is_pdf_preview": False, "download_button_coordinates": None, "reason": f"Error: {str(e)}"}

    async def _is_valid_pdf(self, path: Path) -> bool:
        """
        Deep validation of PDF file.
        1. Exists and size > 1KB
        2. Magic number header is %PDF
        """
        if not path.exists():
            return False
        if path.stat().st_size < 1000: # < 1KB is definitely not a paper
            return False
            
        try:
            with open(path, "rb") as f:
                header = f.read(5)
                # Check for standard PDF header
                if header.startswith(b'%PDF-'):
                    return True
                # Some PDFs might have garbage before header, but rare in direct downloads
                # Strict check avoids false positives
                return False
        except:
            return False

    async def _save_pdf_response(self, response: Response, path: Path) -> bool:
        """
        Save response body to file with strict Content-Type checking.
        """
        try:
            # 1. Pre-flight Check: Content-Type
            ct = response.headers.get("content-type", "").lower()
            url = response.url.lower()
            
            # Allow: application/pdf, binary/octet-stream (sometimes used for downloads)
            # Block: text/html, application/json
            is_pdf_type = "application/pdf" in ct or "binary/octet-stream" in ct
            is_pdf_ext = url.endswith(".pdf")
            
            if not (is_pdf_type or is_pdf_ext):
                if "text/html" in ct:
                    return False
            
            # 2. Save Data
            data = await response.body()
            
            # 3. Post-save Check: Magic Bytes (in memory before writing to disk is better, but here we write first)
            if not data.startswith(b'%PDF-'):
                return False
                
            with open(path, "wb") as f:
                f.write(data)
                
            # 4. Final Verification
            return await self._is_valid_pdf(path)
            
        except Exception as e:
            print(f"Error saving PDF body: {e}")
            if path.exists(): os.remove(path)
            return False

    async def run(self, papers: List[PaperInfo]):
        async with async_playwright() as p:
            # Launch with headless=False so user can login manually
            print("\n>>> Launching browser for UCAS WebVPN download...")
            browser = await p.chromium.launch(channel="msedge", headless=False)
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
                accept_downloads=True
            )
            
            page = await context.new_page()
            
            # 1. Login Phase (WebVPN)
            print(f"Navigating to WebVPN login: {WEBVPN_LOGIN_URL}")
            use_webvpn = True
            try:
                await page.goto(WEBVPN_LOGIN_URL, timeout=15000)
                print("❗❗❗ PLEASE LOGIN TO UCAS WEBVPN MANUALLY IN THE BROWSER ❗❗❗")
                print("Waiting for successful login...")
                
                # Wait for user to login. 
                # Strategy: Loop and check for elements that only exist after login (e.g., logout button, resource list)
                logged_in = False
                start_time = time.time()
                
                while time.time() - start_time < 300: # Wait up to 5 minutes
                    try:
                        # Check URL changes or specific elements
                        # WebVPN usually redirects to a portal or shows a "Logout" button
                        if "logout" in page.url or "portal" in page.url or "index" in page.url:
                            # Check content to be sure
                            content = await page.content()
                            if "注销" in content or "Logout" in content or "资源列表" in content:
                                print("✅ Login detected! Proceeding...")
                                logged_in = True
                                break
                                
                        await asyncio.sleep(2)
                    except Exception as e:
                        print(f"Waiting for login... ({e})")
                        await asyncio.sleep(2)
                
                if not logged_in:
                    print("❌ Login timeout. Please try again.")
                    return
            except Exception as e:
                print(f"⚠️ WebVPN connection failed: {e}")
                print("⚠️ Switching to Direct Access mode (Will try to download OA papers directly).")
                use_webvpn = False

            # Wait a bit for session to stabilize
            await page.wait_for_timeout(3000)

            # 2. Process Papers
            total = len(papers)
            print(f"Starting processing of {total} papers (Mode: {'WebVPN' if use_webvpn else 'Direct'})...")
            
            for i, paper in enumerate(papers):
                print(f"\n[{i+1}/{total}] Processing: {paper.title[:50]}...")
                
                safe_title = re.sub(r'[\\/*?:"<>|]', "", paper.title)[:150].strip()
                pdf_path = self.download_dir / f"{safe_title}.pdf"
                
                if pdf_path.exists():
                    print("Skipping: Already downloaded")
                    paper.download_status = "Success"
                    paper.local_path = str(pdf_path)
                    continue

                # Open new page for each paper to keep session clean
                paper_page = await context.new_page()
                
                # --- URL Conversion Logic ---
                target_url = paper.link
                # 智能判断：如果用户已经在 WebVPN 环境下（通过转换），则使用转换后的 URL
                # 如果是 OA 论文，可能不需要，但走了 WebVPN 也不影响
                if use_webvpn:
                    webvpn_url = convert_to_webvpn_url(target_url)
                    print(f"  Target URL: {target_url}")
                    print(f"  WebVPN URL: {webvpn_url}")
                    nav_url = webvpn_url
                else:
                    print(f"  Target URL: {target_url} (Direct Access)")
                    nav_url = target_url
                
                download_event = asyncio.Event()
                download_obj = None

                async def handle_download(download):
                    nonlocal download_obj
                    download_obj = download
                    download_event.set()
                
                paper_page.on("download", handle_download)
                
                try:
                    # Navigate to the WebVPN converted link
                    print(f"Navigating to: {nav_url}")
                    response = None
                    try:
                        response = await paper_page.goto(nav_url, wait_until="domcontentloaded", timeout=60000)

                        # Explicit wait for potential redirects/rendering as requested by user
                        print("Waiting for page content to fully load (polling for 10s)...")
                        # Polling wait: Check if URL changes to PDF or if page indicates loading
                        start_wait = time.time()
                        while time.time() - start_wait < 10:
                            if paper_page.url.lower().endswith(".pdf"):
                                print("  > URL changed to .pdf extension!")
                                break
                            await asyncio.sleep(1)
                        
                        print(f"Current URL after wait: {paper_page.url}")
                        
                    except Exception as e:
                        if "Download is starting" in str(e) or "net::ERR_ABORTED" in str(e):
                            print(f"Navigation triggered download: {e}")
                        else:
                            raise e

                    # 1. Check for Attachment Download
                    try:
                        await asyncio.wait_for(download_event.wait(), timeout=3.0)
                    except asyncio.TimeoutError:
                        pass
                        
                    if download_obj:
                        print(f"Download started (Attachment): {download_obj.suggested_filename}")
                        await download_obj.save_as(pdf_path)
                        
                        if await self._is_valid_pdf(pdf_path):
                            print(f"✅ Saved to: {pdf_path}")
                            paper.download_status = "Success"
                            paper.local_path = str(pdf_path)
                            await paper_page.close()
                            continue
                        else:
                            print(f"❌ Downloaded file invalid (corrupted/not PDF). Deleting...")
                            if pdf_path.exists(): os.remove(pdf_path)
                            download_obj = None

                    # 2. Check for Inline PDF Response
                    if response:
                        # Retry logic for direct PDF save
                        # Sometimes the first response is HTML (redirect), but subsequent requests might be PDF
                        # OR the page url itself is now PDF
                        
                        target_response = response
                        # Check if current page URL is better
                        if paper_page.url.lower().endswith(".pdf") and paper_page.url != response.url:
                             print("  > URL mismatch. Page URL is PDF. Trying to fetch from page URL...")
                             pass

                        ct = target_response.headers.get("content-type", "").lower()
                        if "application/pdf" in ct or paper_page.url.lower().endswith(".pdf"):
                            print("Detected direct PDF content (via Header or URL). Saving...")
                            if await self._save_pdf_response(target_response, pdf_path):
                                print(f"✅ Saved to: {pdf_path}")
                                paper.download_status = "Success"
                                paper.local_path = str(pdf_path)
                            else:
                                # Failed validation (maybe it was HTML "Loading..." page)
                                print("  > Direct save failed validation. Page might be a viewer. Falling back to VLM...")
                                pass
                            
                            if paper.download_status == "Success":
                                await paper_page.close()
                                continue

                    # 3. HTML Page -> VLM Analysis (Landing Page Logic)
                    if paper_page.is_closed():
                        paper.download_status = "Failed"
                        paper.fail_reason = "Page closed unexpectedly"
                        continue

                    await paper_page.wait_for_timeout(3000) # Wait for render
                    
                    print("Analyzing landing page with Vision Model...")
                    screenshot = await paper_page.screenshot(type="png")
                    b64_img = base64.b64encode(screenshot).decode('utf-8')
                    
                    analysis = await self.analyze_page(b64_img)
                    
                    # 优先检查验证码
                    if analysis.get("verification_present"):
                        print("❌ Verification/Loading detected (Wait state).")
                        # Should we wait more? 
                        # If it says "Please wait...", we should loop.
                        # For now, let's mark failed but distinguish reason
                        paper.fail_reason = "需要人机验证或正在加载中"
                        # If simple loading, maybe retry?
                        # Implementing simple 1-time retry if "wait" is detected
                        if "wait" in analysis.get("reason", "").lower():
                             print("  > VLM suggests waiting. Sleeping 10s and retrying VLM...")
                             await asyncio.sleep(10)
                             # Retry VLM once
                             screenshot = await paper_page.screenshot(type="png")
                             b64_img = base64.b64encode(screenshot).decode('utf-8')
                             analysis = await self.analyze_page(b64_img)
                    
                    # Check if VLM thinks it IS a PDF preview
                    if analysis.get("is_pdf_preview"):
                          print("✅ VLM detected PDF Preview interface! Attempting shortcut Save...")
                          
                          # Try Ctrl+S first (Standard Browser Save)
                          # This often triggers the 'download' event in Playwright
                          download_event.clear()
                          download_obj = None
                          try:
                              await paper_page.keyboard.press("Control+s")
                              # Wait short time for trigger
                              start_wait = time.time()
                              while time.time() - start_wait < 5:
                                  if download_obj: break
                                  await asyncio.sleep(0.5)
                                  
                              if download_obj:
                                  print(f"Ctrl+S triggered download: {download_obj.suggested_filename}")
                                  await download_obj.save_as(pdf_path)
                                  if await self._is_valid_pdf(pdf_path):
                                      print(f"✅ Saved to: {pdf_path}")
                                      paper.download_status = "Success"
                                      paper.local_path = str(pdf_path)
                                      await paper_page.close()
                                      continue
                          except Exception as e:
                              print(f"Ctrl+S failed: {e}")
                          pass

                    if analysis.get("verification_present"): # Re-check after retry
                        print("❌ Verification/Loading persists.")
                        paper.download_status = "Failed"
                        paper.fail_reason = "验证/加载超时"
                        await paper_page.close()
                        continue
                        
                    coords = analysis.get("download_button_coordinates")
                    if coords:
                        ymin, xmin, ymax, xmax = coords
                        width = 1280
                        height = 800
                        x = (xmin + xmax) / 2 / 1000 * width
                        y = (ymin + ymax) / 2 / 1000 * height
                        
                        print(f"Found PDF button at ({x:.1f}, {y:.1f}). Clicking...")
                        
                        # Reset download event for click
                        download_event.clear()
                        download_obj = None
                        
                        # 优化点击策略：不强制等待导航完成，而是给予超长等待时间监听下载事件
                        # 因为点击 PDF 链接可能直接触发下载（不跳转），也可能跳转加载 PDF（很慢）
                        try:
                            await paper_page.mouse.click(x, y)
                        except Exception as e:
                            print(f"Click failed: {e}")
                        
                        # 轮询等待下载，最长等待 90 秒
                        print("Waiting for download to trigger (timeout 90s)...")
                        wait_start = time.time()
                        downloaded = False
                        
                        while time.time() - wait_start < 90:
                            if download_obj:
                                print(f"Click triggered download: {download_obj.suggested_filename}")
                                await download_obj.save_as(pdf_path)
                                
                                if await self._is_valid_pdf(pdf_path):
                                    print(f"✅ Saved to: {pdf_path}")
                                    paper.download_status = "Success"
                                    paper.local_path = str(pdf_path)
                                    downloaded = True
                                    break
                                else:
                                    print("❌ Click downloaded invalid file. Retrying wait...")
                                    if pdf_path.exists(): os.remove(pdf_path)
                                    download_obj = None
                            
                            await asyncio.sleep(1)
                        
                        if not downloaded:
                            print("❌ Timeout waiting for download after click.")
                            paper.download_status = "Failed"
                            paper.fail_reason = "点击后超时/无响应"
                            
                    else:
                        print("❌ No PDF button found.")
                        paper.download_status = "Failed"
                        paper.fail_reason = "未找到下载按钮"

                except Exception as e:
                    print(f"❌ Error: {e}")
                    paper.download_status = "Failed"
                    if "Vision analysis failed" in str(e):
                         paper.fail_reason = "VLM分析失败"
                    else:
                         paper.fail_reason = f"运行错误: {str(e)[:50]}..."
                finally:
                    if not paper_page.is_closed():
                        await paper_page.close()
            
            print("\nAll tasks completed.")
            await browser.close()

async def run_download_workflow(scored_papers: List[PaperInfo], output_dir: Path):
    """
    Exposed function for integrated workflow.
    Takes a list of scored papers, filters for high scores, and runs download.
    """
    pdfs_dir = output_dir / "pdfs"
    report_path = output_dir / "download_report.md"
    
    # Filter for download candidates
    target_papers = [p for p in scored_papers if p.score >= 60]
    print(f"Found {len(target_papers)} papers with score >= 60.")
    
    if not target_papers:
        print("No papers to download.")
        return

    # Run Downloader
    downloader = SEPDownloader(str(pdfs_dir))
    await downloader.run(target_papers)
    
    # Generate Report
    generate_report(target_papers, str(report_path))

def generate_report(papers: List[PaperInfo], output_path: str):
    """Generates a markdown report of SEP download status."""
    print(f"\nGenerating SEP download report at: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# UCAS SEP 自动下载报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计处理: {len(papers)} 篇\n")
        
        success_count = sum(1 for p in papers if p.download_status == "Success")
        failed_count = sum(1 for p in papers if p.download_status == "Failed")
        
        f.write(f"- ✅ 成功: {success_count}\n")
        f.write(f"- ❌ 失败: {failed_count}\n\n")
        
        f.write("---\n\n")
        
        # Section 1: Successful
        f.write("## ✅ 已成功下载\n")
        success_papers = [p for p in papers if p.download_status == "Success"]
        for p in success_papers:
            f.write(f"- **[{p.id}] {p.title}**\n")
            f.write(f"  - 路径: `{p.local_path}`\n")
            
        f.write("\n---\n\n")

        # Section 2: Failed
        f.write("## ❌ 下载失败\n")
        failed_papers = [p for p in papers if p.download_status == "Failed"]
        if not failed_papers:
            f.write("*无失败文献*\n")
            
        for p in failed_papers:
            f.write(f"### {p.id}. {p.title}\n")
            f.write(f"- **链接**: [{p.link}]({p.link})\n")
            f.write(f"- **失败原因**: {p.fail_reason}\n\n")

async def main():
    # Standalone mode: Prefer reading from candidates_scored.jsonl
    
    base_dir = Path(project_root) / "langchain_version" / "examples" / "download_test_merged"
    
    jsonl_path = base_dir / "candidates_scored.jsonl"
    md_path = base_dir / "literature_review_scored.md"
    fallback_md_path = base_dir / "literature_review_with_downloads.md"
    
    download_dir = base_dir / "pdfs"
    report_path = base_dir / "sep_download_report.md"
    
    parser = LiteratureParser("") # dummy init
    all_papers = []
    
    if jsonl_path.exists():
        print(f"Reading papers from JSONL: {jsonl_path}")
        all_papers = parser.parse_jsonl(str(jsonl_path))
    elif md_path.exists():
        print(f"Reading papers from Scored Markdown: {md_path}")
        parser.md_path = str(md_path)
        all_papers = parser.parse()
    elif fallback_md_path.exists():
        print(f"Reading papers from Fallback Markdown: {fallback_md_path}")
        parser.md_path = str(fallback_md_path)
        all_papers = parser.parse()
    else:
        print(f"No input file found in {base_dir}")
        return
    
    # Filter for Score >= 60
    target_papers = [p for p in all_papers if p.score >= 60]
            
    print(f"Found {len(target_papers)} papers needing download (Score >= 60).")
    if not target_papers:
        print("No papers to download. Exiting.")
        return

    downloader = SEPDownloader(str(download_dir))
    await downloader.run(target_papers)
    
    generate_report(target_papers, str(report_path))

if __name__ == "__main__":
    asyncio.run(main())
