"""AI が特定した条文番号を unique_anchor に解決するモジュール。"""

import logging
import re

logger = logging.getLogger(__name__)


def resolve_to_anchors(
    mentioned_articles: list[str], articles: list
) -> tuple[set[str], list[tuple[str, object]]]:
    """AI が出力した条文番号リストを articles 内の unique_anchor に解決する。

    Args:
        mentioned_articles: AI が返した条文番号文字列のリスト
            例: ["1044", "1044の2", "附則1", "3"]
        articles: BQ から取得した ArticleWithSummary のリスト

    Returns:
        (anchors, matched_pairs):
            anchors: 解決された unique_anchor の set
            matched_pairs: (display_str, article) のリスト（照合情報プレフィックス生成用）
    """
    if not mentioned_articles or not articles:
        return set(), []

    anchors: set[str] = set()
    matched_pairs: list[tuple[str, object]] = []

    for raw in mentioned_articles:
        suffix, display, prefix = _parse_article_number(raw)
        if not suffix:
            logger.warning(f"Could not parse AI mentioned article: '{raw}'")
            continue

        pattern = re.compile(rf"{prefix}Article_{suffix}$")
        for article in articles:
            ua = getattr(article, "unique_anchor", "")
            if pattern.search(ua):
                anchors.add(ua)
                matched_pairs.append((display, article))
                break
        else:
            fallback_anchor = f"{prefix}Article_{suffix}"
            anchors.add(fallback_anchor)
            logger.info(f"AI mentioned article '{raw}' not found, using: {fallback_anchor}")

    logger.info(f"Resolved {len(anchors)} mentioned anchors from AI: {anchors}")
    return anchors, matched_pairs


def build_prefix_from_resolved(
    matched_pairs: list[tuple[str, object]], query: str
) -> str:
    """解決済み条文ペアから照合情報プレフィックスを生成する。"""
    if not matched_pairs:
        return ""

    has_paragraph_or_item = bool(
        re.search(r"(?:第\d+項|第\d+号|\d+項|\d+号)", query)
    )

    lines = [
        "【クエリで指定された条文の照合情報 - 回答前に必ず確認すること】",
        "クエリに以下の条文番号が含まれています。この情報と照合した上で、前提が誤っている場合は冒頭で訂正してください。",
        "",
    ]
    for display, article in matched_pairs:
        summary = getattr(article, "article_summary", None) or ""
        lines.append(f"■ {display}の正式タイトル: {summary}")

    if has_paragraph_or_item:
        lines.append("")
        lines.append(
            "※ 本システムは条単位で条文を保持しているため、"
            "指定された項・号を含む条全体を参照しています。"
        )

    lines += ["", "---", ""]
    return "\n".join(lines)


def _parse_article_number(raw: str) -> tuple[str, str, str]:
    """AI出力の条文番号文字列を (anchor_suffix, display_str, anchor_prefix) に変換する。

    入力例と出力:
        "1044"     → ("1044", "第1044条", "Main_")
        "1044の2"  → ("1044_2", "第1044条の2", "Main_")
        "附則1"    → ("1", "附則第1条", "Suppl_")
        "附則3の2" → ("3_2", "附則第3条の2", "Suppl_")

    Returns:
        (suffix, display, prefix) or ("", "", "") if parse failed
    """
    raw = raw.strip()
    if not raw:
        return "", "", ""

    raw = raw.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    raw = raw.replace("第", "").replace("条", "")

    is_suppl = "附則" in raw
    if is_suppl:
        raw = raw.replace("附則", "").strip()

    parts = re.split(r"の", raw)
    nums = [p.strip() for p in parts if p.strip()]
    if not nums:
        return "", "", ""

    for n in nums:
        if not re.match(r"^\d+$", n):
            return "", "", ""

    suffix = "_".join(nums)
    prefix = "Suppl_" if is_suppl else "Main_"

    if is_suppl:
        display = f"附則第{nums[0]}条"
        if len(nums) > 1:
            display += "".join(f"の{n}" for n in nums[1:])
    else:
        display = f"第{nums[0]}条"
        if len(nums) > 1:
            display += "".join(f"の{n}" for n in nums[1:])

    return suffix, display, prefix
