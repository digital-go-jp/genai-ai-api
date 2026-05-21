"""
BigQueryを利用した多段階検索戦略の実装。
"""

import logging
import os

import google.auth
from google.cloud import bigquery
from pydantic import BaseModel, HttpUrl

# ---------------------------
# データモデル (新しいスキーマ)
# ---------------------------


class LawCandidate(BaseModel):
    law_num: str
    law_title: str
    score: float


class ArticleWithSummary(BaseModel):
    law_num: str
    law_id: str
    law_title: str
    unique_anchor: str  # 一意キー
    anchor: str | None = None  # e-Govリンク用アンカー
    article_summary: str | None = None
    content: str | None = None  # SQLで動的に選択されたコンテンツ（summary or content）
    is_summary_only: bool = False  # 100k文字超の法令のため summary のみ返却されたフラグ


class FullArticle(BaseModel):
    law_id: str
    title: str
    content: str
    unique_anchor: str  # 一意キー
    anchor: str | None = None  # e-Govリンク用アンカー
    url: HttpUrl | str


# ---------------------------
# BigQueryリトリーバー
# ---------------------------


class BigQueryRetriever:
    def __init__(self, project: str, dataset: str):
        self.client = bigquery.Client(project=project)
        self.project = project
        self.dataset = dataset
        self.master_table = f"`{project}.{dataset}.app_laws_master`"
        self.indexing_table = f"`{project}.{dataset}.app_laws_for_indexing`"
        self.model_ref = f"`{project}.{dataset}.embedding_model`"
        logging.info(f"BigQueryRetriever initialized for dataset {project}.{dataset}")

    def search_by_law_names(self, law_names: list[str], k: int = 100) -> list[LawCandidate]:
        """
        複数の法令名候補を受け取り、ベクトル検索で類似する法令を検索する。
        """
        if not law_names:
            return []

        query = f"""
            SELECT
                base.law_num,
                base.law_title,
                distance AS score
            FROM
                VECTOR_SEARCH(
                    TABLE {self.master_table},
                    'law_title_embedding',
                    (
                        SELECT ml_generate_embedding_result
                        FROM ML.GENERATE_EMBEDDING(
                            MODEL {self.model_ref},
                            (SELECT * FROM UNNEST(@law_names) AS content)
                        )
                    ),
                    top_k => {k},
                    distance_type => 'COSINE'
                )
            ORDER BY distance ASC, law_num ASC  -- 安定したソート順序
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("law_names", "STRING", law_names),
            ]
        )

        try:
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            return [LawCandidate(**row) for row in results]
        except Exception as e:
            logging.error(f"法令名検索中にエラーが発生: {e}")
            return []

    def get_articles_with_summaries(self, law_nums: list[str]) -> list[ArticleWithSummary]:
        """
        法令番号のリストを受け取り、それらに紐づく全ての条文の概要と一意なアンカーを取得する。
        """
        if not law_nums:
            return []

        query = rf"""
            SELECT
                law_num,
                law_id,
                law_title,
                unique_anchor,
                anchor,
                article_summary
            FROM {self.indexing_table}
            WHERE law_num IN UNNEST(@law_nums)
            ORDER BY law_num,
                     CAST(REGEXP_EXTRACT(unique_anchor, r'Article_(\d+)') AS INT64),
                     unique_anchor
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("law_nums", "STRING", law_nums),
            ]
        )
        try:
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            return [ArticleWithSummary(**row) for row in results]
        except Exception as e:
            logging.error(f"条文概要の取得中にエラーが発生: {e}")
            return []

    def get_articles_by_nearest_law(self, law_names: list[str]) -> list[ArticleWithSummary]:
        """
        法令名候補それぞれから最近傍の法令を特定し、文字数に応じてsummaryまたはcontentを返す。
        各法令名に対して1つずつ最近傍法令を取得し、各法令について
        10万文字を超える場合はsummary、そうでなければcontentを返す。
        """
        if not law_names:
            return []

        query = rf"""
            WITH AllNearestLaws AS (
                SELECT DISTINCT base.law_num
                FROM VECTOR_SEARCH(
                    TABLE {self.master_table},
                    'law_title_embedding',
                    (
                        SELECT ml_generate_embedding_result
                        FROM ML.GENERATE_EMBEDDING(
                            MODEL {self.model_ref},
                            (SELECT * FROM UNNEST(@law_names) AS content)
                        )
                    ),
                    top_k => 1,  -- 各法令名に対して1つずつ最近傍を取得
                    distance_type => 'COSINE'
                )
            ),
            LawContentSize AS (
                SELECT
                    law_num,
                    SUM(CHAR_LENGTH(COALESCE(content, article_summary, '')))
                        > 100000 as is_large_content
                FROM {self.indexing_table}
                WHERE law_num IN (SELECT law_num FROM AllNearestLaws)
                GROUP BY law_num
            )
            SELECT
                indexing.law_num,
                indexing.law_id,
                indexing.law_title,
                indexing.unique_anchor,
                indexing.anchor,
                indexing.article_summary,
                -- 10万文字を超える場合はsummary、そうでなければcontentを返す
                IF(size.is_large_content,
                   indexing.article_summary,
                   COALESCE(indexing.content, indexing.article_summary)) as content,
                size.is_large_content AS is_summary_only
            FROM {self.indexing_table} indexing
            INNER JOIN AllNearestLaws nearest ON indexing.law_num = nearest.law_num
            INNER JOIN LawContentSize size ON indexing.law_num = size.law_num
            ORDER BY indexing.law_num,
                     CAST(REGEXP_EXTRACT(indexing.unique_anchor, r'Article_(\d+)') AS INT64),
                     indexing.unique_anchor
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("law_names", "STRING", law_names),
            ]
        )

        try:
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            return [ArticleWithSummary(**row) for row in results]
        except Exception as e:
            logging.error(f"最近傍法令の条文概要取得中にエラーが発生: {e}")
            return []

    def get_full_articles(
        self, law_nums: list[str], unique_anchors: list[str]
    ) -> list[FullArticle]:
        """
        法令番号と一意なアンカーのリストを受け取り、条文の全文を取得する。
        """
        if not unique_anchors or not law_nums:
            return []

        query = f"""
            SELECT
                law_id,
                law_title || ' ' || COALESCE(article_summary, '') AS title,
                COALESCE(content, '') AS content,
                unique_anchor,
                anchor,
                CASE
                    WHEN anchor IS NOT NULL THEN
                        'https://laws.e-gov.go.jp/law/'
                        || SPLIT(law_id, '_')[OFFSET(0)] || '#' || anchor
                    ELSE
                        'https://laws.e-gov.go.jp/law/'
                        || SPLIT(law_id, '_')[OFFSET(0)]
                END AS url
            FROM {self.indexing_table}
            WHERE law_num IN UNNEST(@law_nums) AND unique_anchor IN UNNEST(@unique_anchors)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("law_nums", "STRING", law_nums),
                bigquery.ArrayQueryParameter("unique_anchors", "STRING", unique_anchors),
            ]
        )
        try:
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()

            full_articles = []
            for row in results:
                try:
                    # 行データをディクショナリに変換
                    row_dict = dict(row)

                    # URLが空またはNoneの場合のフォールバック
                    if not row_dict.get("url"):
                        row_dict["url"] = (
                            f"https://laws.e-gov.go.jp/law/{row_dict.get('law_id', '')}"
                        )

                    # 必須フィールドのデフォルト値設定
                    if not row_dict.get("title"):
                        row_dict["title"] = f"法令 {row_dict.get('law_id', 'unknown')}"
                    if not row_dict.get("content"):
                        row_dict["content"] = "内容が取得できませんでした。"
                    if not row_dict.get("unique_anchor"):
                        row_dict["unique_anchor"] = f"anchor_{len(full_articles)}"

                    full_articles.append(FullArticle(**row_dict))
                except Exception as row_error:
                    logging.warning(f"条文の個別処理でエラー: {row_error}, row: {dict(row)}")
                    # 個別の行でエラーが発生しても処理を続行
                    continue

            return full_articles
        except Exception as e:
            logging.error(f"条文全文の取得中にエラーが発生: {e}")
            return []


# ---------------------------
# グローバルインスタンス
# ---------------------------
retriever: BigQueryRetriever | None = None


def initialize_retriever():
    """
    環境変数から設定を読み込み、リトリーバーのグローバルインスタンスを初期化する。
    """
    global retriever
    if retriever is None:
        try:
            _, project_id = google.auth.default()
        except google.auth.exceptions.DefaultCredentialsError as err:
            raise ConnectionError("Google Cloudの認証情報が見つかりません。") from err

        dataset_id = os.environ.get("BIGQUERY_DATASET")
        if not dataset_id:
            raise ValueError("環境変数 BIGQUERY_DATASET が設定されていません。")

        retriever = BigQueryRetriever(project=project_id, dataset=dataset_id)
        logging.info(f"BigQueryRetrieverが正常に初期化されました (Project: {project_id})。")
