import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import time
from typing import List, Dict, Any

# 禁用安全请求警告 (仅用于解决特定网络环境下的 SSL 握手问题)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 地球物理/地质领域高水平期刊 ID (OpenAlex)
# 涵盖 Nature/Science, JGR, GRL, EPSL 以及主要的地质/石油/构造/地球物理 Q1-Q3 期刊
TOP_JOURNALS = {
    # --- Top Tier (综合与顶刊) ---
    "Nature": "S137773608",
    "Science": "S202381698",
    "Nature Geoscience": "S10021206",
    "PNAS": "S149455428",
    "Science Advances": "S259642502",
    "Nature Communications": "S4210185012",
    
    # --- Geophysics & Tectonics (地球物理与构造) ---
    "GRL": "S36624081",
    "JGR-SE": "S4210228715",
    "EPSL": "S119230507",
    "Tectonophysics": "S195250905",
    "Geology": "S56162041",
    "Geophysics": "S127670868", # SEG
    "GJI": "S108821158", # Geophysical Journal International
    "PEPI": "S106840945", # Physics of the Earth and Planetary Interiors
    "J. Geodynamics": "S140409716",
    "SRL": "S183957208", # Seismological Research Letters
    "BSSA": "S59098468", # Bulletin of the Seismological Society of America
    "Interpretation": "S4210239846",
    
    # --- Regional & General Geology (区域地质与综合地质) ---
    "Precambrian Research": "S1025118869",
    "Gondwana Research": "S158567263",
    "Lithos": "S107629180",
    "JSG": "S13481170", # Journal of Structural Geology
    "JAES": "S80045625", # Journal of Asian Earth Sciences
    "Global Planetary Change": "S63048404",
    "GSA Bulletin": "S3271665",
    "Terra Nova": "S98592585",
    "Geological Magazine": "S2203401",
    "JGS": "S53451518", # Journal of the Geological Society
    "IGR": "S65039790", # International Geology Review
    
    # --- Sedimentary & Petroleum (沉积与石油) ---
    "AAPG Bulletin": "S144020083",
    "Basin Research": "S36808561",
    "Marine Geology": "S5042006",
    "Sedimentary Geology": "S85417211",
    "Marine and Petroleum Geology": "S93565542"
}

class OpenAlexSearcher:
    def __init__(self):
        self.base_url = "https://api.openalex.org/works"
        self.headers = {
            "User-Agent": "EarthPhysicsAgent/2.0 (mailto:researcher@example.com)"
        }
        
        # 配置重试策略
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _reconstruct_abstract(self, inverted_index: Dict[str, List[int]]) -> str:
        """将 OpenAlex 的倒排索引还原为摘要文本"""
        if not inverted_index:
            return ""
        
        # 创建一个 (position, word) 的列表
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        # 按位置排序
        word_positions.sort(key=lambda x: x[0])
        
        # 拼接单词
        return " ".join([word for _, word in word_positions])

    def get_source_id(self, source_name: str) -> str:
        """根据期刊名称获取 OpenAlex ID (Source ID)"""
        try:
            params = {"search": source_name, "per-page": 1}
            response = self.session.get("https://api.openalex.org/sources", params=params, headers=self.headers, timeout=10, verify=False)
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    # 简单匹配：返回第一个结果的 ID
                    # 更严格的匹配可以比较 display_name
                    return results[0]["id"]
        except Exception as e:
            print(f"  [Warning] Resolve source ID for '{source_name}' failed: {e}")
        return None

    def search(self, queries: List[str], negative_terms: List[str] = None, limit_per_query: int = 40, extra_filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """执行多策略混合搜索并返回标准化结果
        
        策略 1: 核心期刊定向检索 (High Precision)
        策略 2: OA 优先检索 (High Accessibility)
        策略 3: 全网关键词检索 (General Recall)
        """
        all_results = {} # 使用 ID 去重
        negative_terms = [t.lower() for t in (negative_terms or [])]
        
        # 基础过滤: 1980年以后, 文章类型
        base_filter_dict = {
            "from_publication_date": "1980-01-01",
            "type": "article"
        }
        if extra_filters:
            base_filter_dict.update(extra_filters)

        # 辅助函数：执行单次请求
        def _execute_request(query_str: str, filters: Dict[str, str], strategy_name: str, limit: int = 20):
            filter_str = ",".join([f"{k}:{v}" for k, v in filters.items()])
            # print(f"    [{strategy_name}] {query_str} (Filter: {filter_str[:30]}...)")
            params = {
                "search": query_str,
                "per-page": limit,
                "filter": filter_str,
                # 综合排序: 相关性 > 引用数
                "sort": "relevance_score:desc,cited_by_count:desc"
            }
            
            # 增加应用层重试机制，应对 SSLError 等底层网络错误
            max_app_retries = 3
            for attempt in range(max_app_retries):
                try:
                    response = self.session.get(self.base_url, params=params, headers=self.headers, timeout=60, verify=False)
                    response.raise_for_status()
                    return response.json().get("results", [])
                except Exception as e:
                    is_last_attempt = (attempt == max_app_retries - 1)
                    if is_last_attempt:
                        print(f"    [Error] {strategy_name} Search Failed after {max_app_retries} attempts: {e}")
                        return []
                    else:
                        time.sleep(1 * (attempt + 1)) # Backoff
            return []

        # 构建 Top Journals 过滤串
        top_venues_filter = "|".join(TOP_JOURNALS.values())

        print(f"  [Search Engine] 开始多策略检索，关键词: {queries}")

        for query in queries:
            # --- 策略 1: 顶级期刊定向检索 (Top Journals) ---
            # 即使不是 OA 也要，因为这些文章质量最高，符合用户对“高水平”的要求
            # 增加权重：这些期刊的结果往往更有价值
            tj_filters = base_filter_dict.copy()
            
            # 关键修正：只有当外部未指定 source.id 时，才使用内置的 TOP_JOURNALS
            # 否则（如 staged search），应尊重外部传入的 source.id
            if "primary_location.source.id" not in tj_filters:
                tj_filters["primary_location.source.id"] = top_venues_filter
            
            # 增加 TopJournal 的召回配额
            results_tj = _execute_request(query, tj_filters, "TopJournal", limit=limit_per_query)
            self._process_results(results_tj, all_results, negative_terms, source_tag="TopJournal")

            # --- 策略 2: OA 优先检索 (OA Only) ---
            # 确保有一部分结果是能下载的
            oa_filters = base_filter_dict.copy()
            oa_filters["open_access.is_oa"] = "true"
            
            results_oa = _execute_request(query, oa_filters, "OA_Only", limit=limit_per_query)
            self._process_results(results_oa, all_results, negative_terms, source_tag="OA_Only")

            # --- 策略 3: 广撒网 (General) ---
            # 补充漏网之鱼
            gen_filters = base_filter_dict.copy()
            results_gen = _execute_request(query, gen_filters, "General", limit=limit_per_query)
            self._process_results(results_gen, all_results, negative_terms, source_tag="General")
            
            time.sleep(0.5) # 避免过快请求

        # 最终确定性排序：优先 TopJournal，其次 Cited Count，最后 ID
        final_results = list(all_results.values())
        
        def sort_key(p):
            # 优先级: TopJournal (2) > OA (1) > General (0)
            # 辅助: 引用数
            priority = 2 if "TopJournal" in p["tags"] else (1 if p["is_oa"] else 0)
            return (priority, p.get("cited_by_count", 0), p["id"])
            
        final_results.sort(key=sort_key, reverse=True)
        
        print(f"  [Search Engine] 检索完成，去重后共 {len(final_results)} 篇文献。")
        return final_results

    def _process_results(self, results, all_results_dict, negative_terms, source_tag):
        """处理原始结果并存入字典"""
        for item in results:
            paper_id = item.get("id")
            
            # 如果已存在，只需更新 tag（如果是更高优先级的 tag）
            if paper_id in all_results_dict:
                if source_tag not in all_results_dict[paper_id]["tags"]:
                    all_results_dict[paper_id]["tags"].append(source_tag)
                continue
                
            title = item.get("title")
            if not title: continue
            
            abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))
            if not abstract or len(abstract) < 50 or "no abstract available" in abstract.lower():
                continue
            
            # 负面词过滤
            text_to_check = (title + " " + abstract).lower()
            if any(neg in text_to_check for neg in negative_terms):
                continue

            # 提取作者
            authors_list = [a.get("author", {}).get("display_name") for a in item.get("authorships", [])]
            if not authors_list: continue
            authors = ", ".join([a for a in authors_list if a][:5])

            # 提取期刊
            primary_loc = item.get("primary_location") or {}
            source = primary_loc.get("source") or {}
            venue = source.get("display_name")
            if not venue: continue

            # 提取链接
            doi = item.get("doi")
            oa = item.get("open_access") or {}
            oa_url = oa.get("oa_url")
            link = oa_url or doi or primary_loc.get("landing_page_url") or paper_id
            
            paper = {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": item.get("publication_year"),
                "link": link,
                "is_oa": oa.get("is_oa", False),
                "source": "OpenAlex",
                "venue": venue,
                "cited_by_count": item.get("cited_by_count", 0),
                "tags": [source_tag] # 标记来源策略
            }
            all_results_dict[paper_id] = paper
