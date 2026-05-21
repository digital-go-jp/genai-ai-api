"""references_text のサイズ管理と段階的 degradation を行うモジュール。"""

import logging

from .retrieval_bq import FullArticle

logger = logging.getLogger(__name__)

MAX_REFERENCES_CHARS = 200_000
PRIORITY_FULL_TEXT_COUNT = 10


def apply_budget(
    final_articles: list[FullArticle],
    mentioned_anchors: set[str],
    max_chars: int = MAX_REFERENCES_CHARS,
) -> list[FullArticle]:
    """
    references_text のサイズが max_chars を超えないよう、段階的に条文を degrade する。

    優先度:
      1. mentioned_anchors（ユーザー指定条文）→ 絶対に全文維持
      2. AI選択の上位 PRIORITY_FULL_TEXT_COUNT 件 → 全文維持
      3. それ以降 → summary に格下げ
      4. それでも超過 → 下位から条文を除外

    Args:
        final_articles: AI選択順に並んだ FullArticle のリスト
        mentioned_anchors: ユーザーが指定した条文の unique_anchor セット（degradation 対象外）
        max_chars: 上限文字数

    Returns:
        budget 内に収まるよう degrade/削減された FullArticle のリスト
    """
    if not final_articles:
        return []

    total_chars = _estimate_total_chars(final_articles)
    if total_chars <= max_chars:
        logger.info(
            f"references_text は予算内です ({total_chars}/{max_chars} chars, "
            f"{len(final_articles)} 条文)"
        )
        return final_articles

    logger.warning(
        f"references_text が予算を超過しています ({total_chars}/{max_chars} chars). "
        "段階的に degradation を試みます。"
    )

    level1_articles = _apply_level1_degradation(
        final_articles, mentioned_anchors, max_chars
    )
    level1_chars = _estimate_total_chars(level1_articles)
    if level1_chars <= max_chars:
        logger.warning(
            f"Level 1 degradation 適用: 下位条文を summary 化 "
            f"({level1_chars}/{max_chars} chars, {len(level1_articles)} 条文)"
        )
        return level1_articles

    level2_articles = _apply_level2_degradation(
        final_articles, mentioned_anchors, max_chars
    )
    level2_chars = _estimate_total_chars(level2_articles)
    if level2_chars <= max_chars:
        logger.warning(
            f"Level 2 degradation 適用: 下位条文を除外 "
            f"({level2_chars}/{max_chars} chars, {len(level2_articles)} 条文)"
        )
        return level2_articles

    level3_articles = _apply_level3_degradation(
        final_articles, mentioned_anchors, max_chars
    )
    level3_chars = _estimate_total_chars(level3_articles)
    logger.warning(
        f"Level 3 degradation 適用: 上位から順に予算内に収める "
        f"({level3_chars}/{max_chars} chars, {len(level3_articles)} 条文)"
    )
    return level3_articles


def _estimate_total_chars(articles: list[FullArticle]) -> int:
    """各条文の content の文字数合計を計算する。"""
    return sum(len(article.content) for article in articles)


def _apply_level1_degradation(
    articles: list[FullArticle], mentioned_anchors: set[str], max_chars: int
) -> list[FullArticle]:
    """
    Level 1: mentioned_anchors と上位 PRIORITY_FULL_TEXT_COUNT 件以外を summary 化。
    """
    result: list[FullArticle] = []

    for idx, article in enumerate(articles):
        is_mentioned = article.unique_anchor in mentioned_anchors
        is_priority = idx < PRIORITY_FULL_TEXT_COUNT

        if is_mentioned or is_priority:
            result.append(article)
        else:
            truncated_content = _truncate_content(article.content, 200)
            result.append(article.model_copy(update={"content": truncated_content}))

    return result


def _apply_level2_degradation(
    articles: list[FullArticle], mentioned_anchors: set[str], max_chars: int
) -> list[FullArticle]:
    """
    Level 2: mentioned_anchors と上位 PRIORITY_FULL_TEXT_COUNT 件以外を除外。
    """
    result: list[FullArticle] = []

    for idx, article in enumerate(articles):
        is_mentioned = article.unique_anchor in mentioned_anchors
        is_priority = idx < PRIORITY_FULL_TEXT_COUNT

        if is_mentioned or is_priority:
            result.append(article)

    return result


def _apply_level3_degradation(
    articles: list[FullArticle], mentioned_anchors: set[str], max_chars: int
) -> list[FullArticle]:
    """
    Level 3: mentioned_anchors 以外を上位から順に追加して max_chars に収める。
    """
    mentioned_articles = [
        article for article in articles if article.unique_anchor in mentioned_anchors
    ]
    other_articles = [
        article for article in articles if article.unique_anchor not in mentioned_anchors
    ]

    result: list[FullArticle] = list(mentioned_articles)
    current_chars = _estimate_total_chars(result)

    for article in other_articles:
        article_chars = len(article.content)
        if current_chars + article_chars <= max_chars:
            result.append(article)
            current_chars += article_chars
        else:
            break

    return result


def _truncate_content(content: str, max_len: int) -> str:
    """content を max_len 文字に切り詰め、末尾に ... を追加する。"""
    if len(content) <= max_len:
        return content
    return content[:max_len] + "..."
