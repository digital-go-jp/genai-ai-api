import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from . import article_resolver, gemini_helpers, prompts, reference_budget
from .gemini_usage_tracker import UsageTracker
from .report_utils import (
    _format_reference,
    _format_reference_for_prompt,
    convert_citation_to_external_link,
    sanitize_mermaid_content,
)
from .retrieval_bq import ArticleWithSummary, FullArticle  # noqa: F401
from .schemas import RequestBody

logger = logging.getLogger(__name__)

_STRICT_CONFIG = {
    "temperature": 0.0,
    "max_output_tokens": 8192,
    "top_p": 1.0,
    "top_k": 1,
    "candidate_count": 1,
}


def _parse_ai_selection(selection_str: str, max_index: int) -> list[int]:
    """AI応答から選択されたインデックスを解析する"""
    selected_indices = []
    try:
        raw_numbers = re.findall(r"\d+", selection_str)
        for num_str in raw_numbers:
            idx = int(num_str)
            if 1 <= idx <= max_index:
                selected_indices.append(idx)
    except Exception as e:
        logger.error(f"AI選択結果の解析エラー: {e}")

    if not selected_indices:
        logger.warning("AI selection parsing failed. Using default selection strategy.")
        return list(range(1, min(max_index + 1, 6)))

    seen: set[int] = set()
    unique = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique[:50]


def _filter_references_by_citations(report_text: str, all_references: list) -> list:
    """レポート内で実際に引用された参照のみを抽出する（元の番号を保持）"""
    raw_nums = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", report_text)
    cited_indices = sorted({int(n.strip()) for group in raw_nums for n in group.split(",")})
    filtered_refs = [
        (i, all_references[i - 1])  # (original_index, reference)
        for i in cited_indices
        if 1 <= i <= len(all_references)
    ]
    return filtered_refs


def _estimate_law_names(
    query, genai_client, app_config, usage_tracker
) -> tuple[list[str], list, list[str]]:
    """法令名を推定して (estimated_law_names, web_hits, mentioned_articles) を返す"""
    # 2. 法令名推定（Web grounding + 3段階フォールバック）
    logger.info("Estimating law names with web grounding and fallback parsing...")
    estimated_law_names = []
    mentioned_articles: list[str] = []

    # クエリにURLが含まれる場合は url_context も有効にしてページ内容を読ませる
    has_url = _URL_IN_QUERY_PATTERN.search(query)
    grounding_tools = "web_search,url_context" if has_url else "web_search"
    if has_url:
        logger.info("URL detected in query. Enabling url_context for law name estimation.")

    try:
        # Web grounding でJSON出力を指示
        logger.info("Web grounding with JSON instruction...")
        today_str = date.today().isoformat()
        url_context_instruction = (
            "クエリにURLが含まれる場合はそのURLの内容を必ず読み取り、内容に基づいて法令名を特定すること。"
            if has_url
            else ""
        )
        grounding_request = RequestBody(
            input_text=f"以下のクエリに関連する日本の法令名を調査して、JSON形式で回答してください。説明文は不要です。JSONのみ出力してください。\n\nクエリ: {query}",
            grounding=grounding_tools,
            system_instruction=f'本日の日付は {today_str} です。{url_context_instruction}クエリに関連する日本の法令を調査し、関連する法令名とクエリが言及している条文番号を以下のJSON形式で回答してください。調査の際はe-Govや各省庁の公式サイト（.go.jpドメイン）を優先して参照してください。必ず有効なJSONのみを出力し、説明文やマークダウンは一切含めないでください：{{"law_names": ["法令名1", "法令名2"], "mentioned_articles": ["1044", "1044の2", "附則3"]}}。mentioned_articlesにはクエリが具体的に言及している条文番号を半角数字で列挙すること。「の」で枝番を表記し、附則の条は「附則」を前置すること。条文番号が言及されていない場合は空配列を返すこと。【重要1】廃止・失効した法令は絶対に含めないこと。本日時点（{today_str}）で既に廃止・統合されている法令は除外し、現行の後継法令名のみを返すこと（例：「行政機関個人情報保護法」は2022年に廃止され「個人情報の保護に関する法律」に統合済みのため、後者を返す）。廃止・改正の有無が不明な場合はe-Govの最新情報を参照して確認すること。【重要2】クエリで言及された法令名が通称・略称・俗称の場合、対応する正式名称が確実に特定できる場合のみ採用すること。正式名称が不明確または実在が確認できない場合はその法令名を含めないこと（存在しない法令を推測で別の法令に読み替えてはならない）。',
            temperature=0.0,
            max_output_tokens=2048,
            top_p=1.0,
            top_k=1,
            candidate_count=1,
            thinking_budget=0 if not has_url else None,
        )
        contents, gen_config = gemini_helpers.prepare_gemini_request(
            request_body=grounding_request, config=app_config, storage_client=None
        )
        response = gemini_helpers.call_gemini_api(
            app_config.model_id, contents, gen_config, genai_client
        )
        usage_tracker.add_usage(response)
        web_hits = gemini_helpers.extract_grounding_web_hits(response, follow_redirects=False)
        logger.info(f"Grounding web hits extracted: {len(web_hits)} results.")
        response_text = response.text

        logger.info(f"Web grounding response length: {len(response_text)} chars")
        logger.info(f"Web grounding response preview: {response_text[:500]}...")

        # Stage 1: 直接JSON解析を試行（```json コードブロックの囲みも除去）
        stripped_text = re.sub(r"^```(?:json)?\s*\n?|```\s*$", "", response_text.strip())
        try:
            result = json.loads(stripped_text)
            estimated_law_names = result.get("law_names", [])
            mentioned_articles = result.get("mentioned_articles", [])
            logger.info(f"Stage 1 success - Direct JSON parsing: {estimated_law_names}")
            logger.info(f"Stage 1 mentioned_articles: {mentioned_articles}")
        except json.JSONDecodeError:
            logger.info("Stage 1 failed - Trying Stage 2: JSON extraction with regex")

            # Stage 2: 正規表現でJSON部分を抽出してから解析
            json_pattern = r'\{[^{}]*"law_names"\s*:\s*\[[^\]]*\][^{}]*\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)

            for json_match in json_matches:
                try:
                    result = json.loads(json_match)
                    estimated_law_names = result.get("law_names", [])
                    mentioned_articles = result.get("mentioned_articles", [])
                    logger.info(f"Stage 2 success - Extracted JSON parsing: {estimated_law_names}")
                    logger.info(f"Stage 2 mentioned_articles: {mentioned_articles}")
                    break
                except json.JSONDecodeError:
                    continue

            # Stage 3: 正規表現で法令名を直接抽出
            if not estimated_law_names:
                logger.info("Stage 2 failed - Trying Stage 3: Direct law name extraction")
                law_patterns = [
                    r"([^。、\n]*(?:人工知能|AI)[^。、\n]*(?:法|規則|省令|政令|条例)[^。、\n]*)",
                    r"([^。、\n]*(?:法|規則|省令|政令|条例)[^。、\n]*)",
                ]

                for pattern in law_patterns:
                    matches = re.findall(pattern, response_text)
                    if matches:
                        # 重複除去と清掃
                        estimated_law_names = list(
                            {
                                match.strip()
                                for match in matches
                                if len(match.strip()) > 3 and len(match.strip()) < 50
                            }
                        )[:10]  # 最大10個まで
                        logger.info(f"Stage 3 success - Regex extraction: {estimated_law_names}")
                        break

            # Stage 4: マークダウン形式（太字・リスト）から法令名を抽出
            if not estimated_law_names:
                logger.info("Stage 3 failed - Trying Stage 4: Markdown law name extraction")
                md_patterns = [
                    r"\*\*([^*]*(?:法律|法|規則|省令|政令|条例)[^*]*)\*\*",
                    r"\*([^*]*(?:法律|法|規則|省令|政令|条例)[^*]*)\*",
                    r"[*\-]\s*(.+?(?:法律|法|規則|省令|政令|条例)[^\n（(]*)",
                ]
                for pattern in md_patterns:
                    matches = re.findall(pattern, response_text)
                    if matches:
                        cleaned = []
                        for m in matches:
                            name = re.sub(r"\s*[\(（].*?[)）]", "", m.strip())
                            name = name.strip(" *-・")
                            if 4 <= len(name) <= 60:
                                cleaned.append(name)
                        if cleaned:
                            estimated_law_names = list(dict.fromkeys(cleaned))[:10]
                            logger.info(
                                f"Stage 4 success - Markdown extraction: {estimated_law_names}"
                            )
                            break

        if not estimated_law_names:
            logger.warning("All stages (1-4) failed to extract law names")
        else:
            logger.info(f"Final extracted law names: {estimated_law_names}")

    except Exception as e:
        logger.error(f"2-stage law name estimation failed: {e}")
        # システムエラーかコンテンツの問題かを判定
        error_str = str(e).lower()
        if any(
            keyword in error_str
            for keyword in ["timeout", "connection", "api", "network", "service", "unavailable"]
        ):
            return ([], [], [])
        else:
            return ([], [], [])

    # 法令名推定が空の結果を返した場合
    if not estimated_law_names:
        logger.warning("Law name estimation returned empty result")
        return ([], [], [])

    return (estimated_law_names, web_hits, mentioned_articles)


def _search_articles(law_names, bq_retriever) -> list:
    """BigQueryで法令条文を検索して返す"""
    # 3. BigQuery 検索（法令条文取得）
    logger.info("Starting BigQuery nearest law articles search...")
    articles = []
    try:
        articles = bq_retriever.get_articles_by_nearest_law(law_names)
        logger.info(f"BigQuery found {len(articles)} articles from nearest law.")

        if not articles:
            logger.warning("No articles found from nearest law. Trying broader search...")
            broader_terms = [*law_names, "法律", "規則", "政令"]
            articles = bq_retriever.get_articles_by_nearest_law(broader_terms)
            logger.info(f"Broader search found {len(articles)} articles.")
    except Exception as e:
        logger.error(f"BigQuery nearest law search failed: {e}")
        articles = []

    return articles


def _select_articles(query, articles, genai_client, app_config, usage_tracker) -> list:
    """AIで関連条文を選択して返す"""
    # 4. AIによる関連条文の選択
    logger.info("AI selecting relevant articles...")
    if len(articles) > 5:  # 5件超の場合のみ選択処理を実行
        try:
            # 条文概要リストを作成
            summary_list_str = "\n".join(
                [
                    f"{i + 1}. {a.law_title} - {a.article_summary if a.article_summary else '概要なし'}"
                    for i, a in enumerate(articles)
                ]
            )

            select_articles_request = RequestBody(
                input_text=f"元のクエリ: {query}\n\n条文概要リスト:\n{summary_list_str}",
                system_instruction=prompts.PROMPT_SELECT_RELEVANT_ARTICLES,
                temperature=_STRICT_CONFIG["temperature"],
                max_output_tokens=_STRICT_CONFIG["max_output_tokens"],
                top_p=_STRICT_CONFIG["top_p"],
                top_k=_STRICT_CONFIG["top_k"],
                candidate_count=_STRICT_CONFIG["candidate_count"],
                thinking_budget=0,
            )
            contents, gen_config = gemini_helpers.prepare_gemini_request(
                request_body=select_articles_request, config=app_config, storage_client=None
            )
            response = gemini_helpers.call_gemini_api(
                app_config.model_id, contents, gen_config, genai_client
            )
            usage_tracker.add_usage(response)

            selected_article_indices = _parse_ai_selection(response.text, len(articles))
            selected_articles = [
                articles[i - 1] for i in selected_article_indices if 1 <= i <= len(articles)
            ]

            if selected_articles:
                articles = selected_articles
                logger.info(
                    f"AI selected {len(articles)} relevant articles from {len(articles)} total"
                )
            else:
                logger.warning("AI article selection failed, using all articles")
        except Exception as e:
            logger.error(f"AI article selection failed: {e}, using all articles")

    return articles


def _to_full_articles(articles) -> list:
    """条文データをFullArticle形式に変換して返す"""
    logger.info("Converting articles to FullArticle format...")
    final_articles = []
    for article in articles:
        if hasattr(article, "content") and article.content:
            law_id_base = article.law_id.split("_")[0]
            anchor = getattr(article, "anchor", None)
            url = (
                f"https://laws.e-gov.go.jp/law/{law_id_base}#{anchor}"
                if anchor
                else f"https://laws.e-gov.go.jp/law/{law_id_base}"
            )
            full_article = FullArticle(
                law_id=article.law_id,
                title=article.law_title,
                content=article.content,
                unique_anchor=article.unique_anchor,
                anchor=anchor,
                url=url,
            )
            final_articles.append(full_article)

    return final_articles


def _build_references(
    final_articles, web_hits, mentioned_anchors: set[str] | None = None
) -> tuple[list, str]:
    """検索結果をマージして参照リストと参考情報テキストを返す"""
    # 5. Web検索結果とBigQuery結果のマージ（条文レベルでの参照対応）
    logger.info("Merging search results...")
    # トークン予算管理: 条文を段階的に degrade して上限内に収める
    article_search_results = reference_budget.apply_budget(
        final_articles, mentioned_anchors or set()
    )
    unique_web_search_results = [
        result
        for result in web_hits
        if not any(article.url == result["url"] for article in final_articles)
    ]

    # マージした結果を作成（条文検索結果 + Web検索結果）
    search_results = [*article_search_results, *unique_web_search_results]

    # Geminiプロンプト用テキスト（URLなし・e-laws条文はラベル付き全文）
    references_text = "\n\n".join(
        [_format_reference_for_prompt(i, r) for i, r in enumerate(search_results)]
    )

    return (search_results, references_text)


_URL_IN_QUERY_PATTERN = re.compile(r"https://\S+")
_ARTICLE_NUM_PATTERN = re.compile(r"第(\d+)条((?:の\d+)*)")

_KANJI_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _kanji_number_to_int(text: str) -> int:
    """漢数字文字列をintに変換する（1〜9999の範囲）。
    例: '十' -> 10, '二十三' -> 23, '百三' -> 103, '千四十四' -> 1044
    """
    result = 0
    current = 0
    for ch in text:
        if ch in _KANJI_DIGITS:
            current = _KANJI_DIGITS[ch]
        elif ch == "千":
            result += (current or 1) * 1000
            current = 0
        elif ch == "百":
            result += (current or 1) * 100
            current = 0
        elif ch == "十":
            result += (current or 1) * 10
            current = 0
    result += current
    return result


_KANJI_ARTICLE_PATTERN = re.compile(
    r"第([一二三四五六七八九十百千]+)条((?:の[一二三四五六七八九十百千]+)*)"
)

_FULLWIDTH_TO_HALFWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")


def _normalize_article_numbers(query: str) -> str:
    """クエリ中の全角数字・漢数字による条番号表記をアラビア数字(半角)に正規化する。
    '第１０条の２' -> '第10条の2', '第十条の十' -> '第10条の10',
    '第二十三条' -> '第23条'
    注: 「第」省略パターン（千四十四条 等）は AI ベース解決に委ねるため対象外。
    """
    query = query.translate(_FULLWIDTH_TO_HALFWIDTH)

    def _replace_kanji(m):
        base = str(_kanji_number_to_int(m.group(1)))
        branch = m.group(2)
        if branch:
            parts = re.findall(r"の([一二三四五六七八九十百千]+)", branch)
            branch = "".join(f"の{_kanji_number_to_int(p)}" for p in parts)
        return f"第{base}条{branch}"

    query = _KANJI_ARTICLE_PATTERN.sub(_replace_kanji, query)

    return query


def _article_match_to_anchor_suffix(match: tuple[str, str]) -> str:
    """正規表現マッチ結果をunique_anchor末尾形式に変換する。
    ('10', 'の10') -> '10_10', ('10', 'の11の2') -> '10_11_2', ('10', '') -> '10'
    """
    base, branch = match
    if branch:
        branch_nums = re.findall(r"\d+", branch)
        return "_".join([base, *branch_nums])
    return base


def _article_match_to_display(match: tuple[str, str]) -> str:
    """正規表現マッチ結果を表示用の条番号に変換する。
    ('10', 'の10') -> '第10条の10'
    """
    base, branch = match
    return f"第{base}条{branch}"


def _build_mentioned_articles_prefix(query: str, articles: list) -> str:
    """クエリで言及された条文番号に対応する条文を抽出し、参考情報の先頭に埋め込むプレフィックスを生成する"""
    query = _normalize_article_numbers(query)
    mentioned_matches = _ARTICLE_NUM_PATTERN.findall(query)
    if not mentioned_matches:
        return ""

    matched = []
    for match in mentioned_matches:
        suffix = _article_match_to_anchor_suffix(match)
        pattern = re.compile(rf"Article_{suffix}$")
        for article in articles:
            if hasattr(article, "unique_anchor") and pattern.search(article.unique_anchor):
                matched.append((match, article))
                break

    if not matched:
        return ""

    has_paragraph_or_item = re.search(r"第\d+条[^。]*?(?:第\d+項|第\d+号)", query)

    lines = [
        "【クエリで指定された条文の照合情報 - 回答前に必ず確認すること】",
        "クエリに以下の条文番号が含まれています。この情報と照合した上で、前提が誤っている場合は冒頭で訂正してください。",
        "",
    ]
    for match, article in matched:
        display = _article_match_to_display(match)
        summary = getattr(article, "article_summary", None) or ""
        lines.append(f"■ {display}の正式タイトル: {summary}")

    if has_paragraph_or_item:
        lines.append("")
        lines.append(
            "※ 本システムは条単位で条文を保持しているため、"
            "指定された項・号を含む条全体を参照しています。"
        )

    lines += ["", "---", ""]

    logger.info(
        f"Mentioned article prefix built for articles: "
        f"{[_article_match_to_display(m[0]) for m in matched]}"
    )
    return "\n".join(lines)


_QUERY_LAW_NAME_PATTERN = re.compile(r"[一-龥ァ-ヴー]+(?:法律|法|規則|政令|条例|省令)")


def _extract_law_names_from_query(query: str) -> list[str]:
    """クエリから法令名候補を直接抽出する（web grounding 前の元の表記を保持するため）"""
    matches = _QUERY_LAW_NAME_PATTERN.findall(query)
    # 短すぎる一般語を除外（「行政法」等、総文字数4未満）
    return [m for m in matches if len(m) >= 4]


def _build_substitution_warning(query_law_names: list[str], estimated_law_names: list[str]) -> str:
    """クエリ元法令名と web grounding 推定名を比較し、読み替えが発生した場合の開示指示を生成する。

    「デジタル行政推進法」→「情報通信技術を活用した行政の推進等に関する法律」のように
    web grounding が暗黙に別の法令へ置き換えた場合、Gemini に読み替えの明示を求める。
    """
    if not query_law_names or not estimated_law_names:
        return ""

    threshold = 0.30

    substituted = []
    for qname in query_law_names:
        sims = {ename: _bigram_similarity(qname, ename) for ename in estimated_law_names}
        best_match = max(sims, key=sims.get)
        best_sim = sims[best_match]
        if best_sim < threshold:
            substituted.append((qname, best_match))
            logger.info(
                f"Query law name substitution detected: '{qname}' → '{best_match}' (sim={best_sim:.2f})"
            )

    if not substituted:
        return ""

    lines = ["【読み替え通知 - 回答の冒頭で必ず開示すること】"]
    for original, replacement in substituted:
        lines.append(
            f"ユーザーが指定した法令名「{original}」は実在しない可能性があります。"
            f"最も近い実在法令「{replacement}」として回答しますが、"
            f"「{original}」が通称・略称、または実在しない法令名である可能性を"
            f"回答の冒頭で明示してください。"
        )
    lines += ["---", ""]
    return "\n".join(lines)


def _expand_law_names_with_ordinances(law_names: list[str]) -> list[str]:
    """法令名リストに施行令・施行規則を補完する。

    本体法名から施行令・施行規則名を生成してBQ検索対象に追加することで、
    本体法に定義規定が少なく政令・省令に委任されている概念（例：個人情報データベース等）の
    条文も取得できるようにする。
    """
    expanded = list(law_names)
    for name in law_names:
        if name.endswith("法律") or name.endswith("法"):
            expanded.append(f"{name}施行令")
            expanded.append(f"{name}施行規則")
    # 重複排除（順序保持）
    seen: set[str] = set()
    result = []
    for n in expanded:
        if n not in seen:
            seen.add(n)
            result.append(n)
    logger.info(f"Expanded law names: {result}")
    return result


def _bigram_similarity(s1: str, s2: str) -> float:
    """バイグラムJaccard係数で2つの法令名の類似度を返す（助詞等を除去して計算）"""
    _particles = re.compile(r"[をにはがのもとでやへからまで等]")
    s1_norm = _particles.sub("", s1)
    s2_norm = _particles.sub("", s2)

    def bigrams(s: str) -> set:
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()

    b1 = bigrams(s1_norm)
    b2 = bigrams(s2_norm)
    if not b1 or not b2:
        return 0.0
    return len(b1 & b2) / len(b1 | b2)


def _check_law_name_divergence(law_names: list[str], articles: list) -> str:
    """推定法令名とBQ取得法令名の乖離を検出し、Geminiへの警告プレフィックスを返す。

    _estimate_law_names が返した名称とBQが実際に取得した law_title を比較し、
    バイグラム類似度が低い（＝名称が大きく異なる）場合に警告テキストを生成する。
    これにより、架空法令名が類似法令にサイレントマッピングされるケースをGeminiに通知できる。
    """
    if not law_names or not articles:
        return ""

    bq_law_titles = list({a.law_title for a in articles if getattr(a, "law_title", None)})
    if not bq_law_titles:
        return ""

    threshold = 0.40

    diverged = []
    for law_name in law_names:
        sims = {title: _bigram_similarity(law_name, title) for title in bq_law_titles}
        best_title = max(sims, key=sims.get)
        best_sim = sims[best_title]
        if best_sim < threshold:
            diverged.append((law_name, best_title, best_sim))
            logger.warning(
                f"Law name divergence detected: '{law_name}' → '{best_title}' (sim={best_sim:.2f})"
            )

    if not diverged:
        return ""

    lines = [
        "【警告】推定された法令名とBQで取得された法令名に大きな乖離があります。",
        "クエリで指定・推定された法令が実在しないか、正式名称と大きく異なる可能性があります。",
        "",
    ]
    for estimated, actual, sim in diverged:
        lines.append(f"- 推定法令名 「{estimated}」 → 取得法令 「{actual}」（類似度: {sim:.0%}）")
    lines += [
        "",
        "この乖離がある場合、回答の冒頭で「指定された法令名が存在しない可能性」または"
        "「通称・略称が正式名称に変換された」旨を明示してください。",
        "---",
        "",
    ]
    return "\n".join(lines)


def _fetch_summary_only_full_content(articles: list, bq_retriever) -> list:
    """_select_articles後の条文リストで、サマリーしか取得できていない条文（100k制限にかかった大きい法令）の全文をBQから取得する"""
    summary_only = [a for a in articles if a.is_summary_only]
    if not summary_only:
        return []

    law_nums = list({a.law_num for a in summary_only if a.law_num})
    unique_anchors = [a.unique_anchor for a in summary_only]

    if not law_nums or not unique_anchors:
        return []

    try:
        full_articles = bq_retriever.get_full_articles(law_nums, unique_anchors)
        logger.info(f"Fetched full content for {len(full_articles)} summary-only articles")
        return full_articles
    except Exception as e:
        logger.error(f"Failed to fetch summary-only full content: {e}")
        return []


def _fetch_mentioned_articles_full_content(query: str, articles: list, bq_retriever) -> list:
    """クエリで言及された条文番号の全文を BQ から直接取得する（100k文字制限を迂回）"""
    query = _normalize_article_numbers(query)
    mentioned_matches = _ARTICLE_NUM_PATTERN.findall(query)
    if not mentioned_matches:
        return []

    law_nums = list({a.law_num for a in articles if a.law_num})
    if not law_nums:
        return []

    unique_anchors = []
    for m in mentioned_matches:
        suffix = _article_match_to_anchor_suffix(m)
        anchor_pattern = re.compile(rf"Article_{suffix}$")
        resolved = None
        for article in articles:
            if hasattr(article, "unique_anchor") and anchor_pattern.search(article.unique_anchor):
                resolved = article.unique_anchor
                break
        unique_anchors.append(resolved or f"Main_Article_{suffix}")

    try:
        full_articles = bq_retriever.get_full_articles(law_nums, unique_anchors)
        logger.info(f"Fetched {len(full_articles)} full articles for mentioned: {unique_anchors}")
        return full_articles
    except Exception as e:
        logger.error(f"Failed to fetch mentioned articles full content: {e}")
        return []


def _generate_complete_report(
    query, references_text, genai_client, app_config, usage_tracker
) -> str:
    """1回でレポート全体を生成して返す"""
    logger.info("Generating complete report in single generation...")

    # クエリに URL が含まれている場合は URL Context ツールを有効化する
    grounding = "url_context" if _URL_IN_QUERY_PATTERN.search(query) else None
    if grounding:
        logger.info("URL detected in query. Enabling URL context tool.")

    # 1回でレポート全体を生成
    complete_report_request = RequestBody(
        input_text=f"クエリ: {query}\n\n参考情報:\n{references_text}",
        grounding=grounding,
        system_instruction=prompts.PROMPT_GENERATE_COMPLETE_REPORT,
        temperature=_STRICT_CONFIG["temperature"],
        max_output_tokens=8192,  # 3,000字程度の出力に対して十分なトークン数
        top_p=_STRICT_CONFIG["top_p"],
        top_k=_STRICT_CONFIG["top_k"],
        candidate_count=_STRICT_CONFIG["candidate_count"],
    )
    contents, gen_config = gemini_helpers.prepare_gemini_request(
        request_body=complete_report_request, config=app_config, storage_client=None
    )
    response = gemini_helpers.call_gemini_api(
        app_config.model_id, contents, gen_config, genai_client
    )
    usage_tracker.add_usage(response)
    report_text = gemini_helpers.extract_text_without_thinking(response)

    # モデルが # 見出し前に出力する thinking 的な前置き文を除去する
    first_heading = re.search(r"^#", report_text, re.MULTILINE)
    if first_heading:
        report_text = report_text[first_heading.start() :]

    return report_text


def _build_url_web_hits(urls: list[str]) -> list[dict]:
    """クエリ中のURLを参考情報エントリとして変換する（ページタイトルをHTTP取得）。
    取得失敗時はURLをそのままタイトルとして使用する。
    """
    if not urls:
        return []

    def _to_hit(url: str) -> dict:
        try:
            after_scheme = url.split("://", 1)[-1]
            fallback_domain = after_scheme.split("/")[0]
            final_url, title = gemini_helpers._fetch_page_info(url, fallback_domain, trusted=False)
            return {"title": title, "url": final_url, "snippet": ""}
        except Exception as e:
            logger.warning(f"URL hit fetch failed for {url}: {e}")
            return {"title": url, "url": url, "snippet": ""}

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(_to_hit, urls))


def _finalize_report(report_text, search_results) -> str:
    """引用リンク変換・Mermaid安全化・参照セクション結合を行い最終レポートを返す"""
    # 参照フィルタリング（実際に引用されたもののみ）
    logger.info("Filtering references based on citations...")
    filtered_references = _filter_references_by_citations(report_text, search_results)

    # フィルタリングに失敗した場合は全参照を使用
    if not filtered_references:
        logger.warning("No citation markers found, using all references as fallback")
        # 全参照を元の番号付きタプル形式で設定
        filtered_references = [(i + 1, ref) for i, ref in enumerate(search_results)]

    # フィルタリングされた参照リストを再生成（元の番号を使用）
    if filtered_references and isinstance(filtered_references[0], tuple):
        # 新しい形式: (original_index, reference) のタプル
        filtered_references_text = "\n\n".join(
            [_format_reference(original_idx - 1, ref) for original_idx, ref in filtered_references]
        )
    else:
        # 旧来の形式（フォールバック用）
        filtered_references_text = "\n\n".join(
            [_format_reference(i, r) for i, r in enumerate(filtered_references)]
        )

    # 本文部分の引用番号を外部URLにリンク化
    temp_final_report_with_links = convert_citation_to_external_link(
        report_text, filtered_references
    )

    # Mermaidコードブロックの安全化処理
    temp_final_report_sanitized = sanitize_mermaid_content(temp_final_report_with_links)

    # 参照セクションは通常の形式（アンカー不要）
    filtered_references_text_with_anchors = filtered_references_text

    # 最終レポート結合
    final_report = "\n\n".join(
        [temp_final_report_sanitized, "## 出典", filtered_references_text_with_anchors]
    )

    actual_ref_count = len(filtered_references) if filtered_references else 0
    logger.info(
        f"Single generation report completed. Using {actual_ref_count} filtered references."
    )

    return final_report


def generate_law_report(
    query: str,
    genai_client,
    app_config,
    bq_retriever,
) -> tuple[str, list[dict]]:
    """法令レポート生成のメイン関数"""
    usage_tracker = UsageTracker()

    # クエリから元の法令名を抽出（web grounding 前の表記を保持）
    query_law_names = _extract_law_names_from_query(query)

    law_names, raw_web_hits, ai_mentioned_articles = _estimate_law_names(
        query, genai_client, app_config, usage_tracker
    )
    if not law_names:
        return (
            "クエリから関連する法令を特定できませんでした。より具体的な法令名（例：民法、刑法、労働基準法など）を含めてクエリを再構成してください。",
            [],
        )

    # 施行令・施行規則を補完してBQ検索対象を拡張
    search_law_names = _expand_law_names_with_ordinances(law_names)

    # BQ 検索・redirect 解決・クエリURL取得を並列実行
    logger.info("Running BQ search, redirect resolution, and URL fetch in parallel...")
    query_urls = _URL_IN_QUERY_PATTERN.findall(query)
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_articles = executor.submit(_search_articles, search_law_names, bq_retriever)
        future_web_hits = executor.submit(gemini_helpers.resolve_redirect_web_hits, raw_web_hits)
        future_url_hits = executor.submit(_build_url_web_hits, query_urls) if query_urls else None
        articles = future_articles.result()
        web_hits = future_web_hits.result()
        url_hits = future_url_hits.result() if future_url_hits else []

    # クエリURLを web_hits の先頭に追加（URL contextの主要ソースを先に番号付与するため）
    if url_hits:
        existing_urls = {h["url"] for h in web_hits}
        web_hits = [h for h in url_hits if h["url"] not in existing_urls] + web_hits
        logger.info(f"Added {len(url_hits)} query URL(s) to web_hits as numbered references")

    if not articles:
        return (
            "申し訳ございませんが、該当する法令が見つかりませんでした。システムの問題が発生している可能性があります。",
            [],
        )

    # クエリ元法令名 vs web grounding 推定名の読み替えチェック
    substitution_warning = _build_substitution_warning(query_law_names, law_names)

    # 推定法令名とBQ取得法令名の乖離チェック（架空法令の誤マッピング検出）
    law_name_divergence_warning = _check_law_name_divergence(law_names, articles)

    articles = _select_articles(query, articles, genai_client, app_config, usage_tracker)

    # AI ベースの条文番号解決（Phase 2）
    if ai_mentioned_articles:
        resolved_anchors, resolved_pairs = article_resolver.resolve_to_anchors(
            ai_mentioned_articles, articles
        )
        mentioned_prefix = article_resolver.build_prefix_from_resolved(resolved_pairs, query)
    else:
        resolved_anchors = set()
        mentioned_prefix = _build_mentioned_articles_prefix(query, articles)

    final_articles = _to_full_articles(articles)
    if not final_articles:
        return "該当する条文が見つかりませんでした。", []

    # 言及条文の全文を BQ から直接取得してマージ（100k文字制限を迂回）
    mentioned_full = _fetch_mentioned_articles_full_content(query, articles, bq_retriever)
    mentioned_anchors: set[str] = resolved_anchors
    if mentioned_full:
        # 既存の same unique_anchor をサマリー版から全文版に差し替え、先頭に配置
        mentioned_anchors = mentioned_anchors | {a.unique_anchor for a in mentioned_full}
        final_articles = mentioned_full + [
            a for a in final_articles if a.unique_anchor not in mentioned_anchors
        ]
        logger.info(f"Prepended {len(mentioned_full)} full-content mentioned articles")

    # サマリーのみ条文（100k制限にかかった大きい法令）の全文を取得してin-place差し替え
    summary_full = _fetch_summary_only_full_content(articles, bq_retriever)
    if summary_full:
        summary_map = {a.unique_anchor: a for a in summary_full}
        final_articles = [summary_map.get(a.unique_anchor, a) for a in final_articles]
        logger.info(f"Upgraded {len(summary_full)} summary-only articles to full content")

    search_results, references_text = _build_references(
        final_articles, web_hits, mentioned_anchors
    )

    # 各種警告・照合情報を参考情報の先頭に埋め込む（優先度順）
    if mentioned_prefix:
        references_text = mentioned_prefix + references_text
    if law_name_divergence_warning:
        references_text = law_name_divergence_warning + references_text
    if substitution_warning:
        references_text = substitution_warning + references_text

    report = _generate_complete_report(
        query,
        references_text,
        genai_client,
        app_config,
        usage_tracker,
    )

    final_report = _finalize_report(report, search_results)

    usage_summary = usage_tracker.get_usage_summary()
    logger.info(f"Usage summary: {usage_summary}")
    return final_report, usage_summary
