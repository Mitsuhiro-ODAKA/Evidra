import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv

load_dotenv()

# --- Azure/OpenAI クライアント初期化 ------------------------------------

def get_aoai_client() -> AzureOpenAI:
    """
    Azure OpenAI クライアントを初期化して返す。
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        raise RuntimeError("Azure OpenAI の環境変数が未設定です。")
    return AzureOpenAI(api_key=key, api_version="2024-06-01", azure_endpoint=endpoint)

def get_cosmos_container():
    """
    Cosmos DB (SQL API) コンテナを初期化して返す。
    コンテナ側は事前にベクトルインデックスを用意しておくのが理想だが、
    フォールバックとして全件取得+アプリ側コサインで近傍検索も可能。
    """
    endpoint = os.getenv("COSMOS_ENDPOINT")
    key = os.getenv("COSMOS_KEY")
    db_name = os.getenv("COSMOS_DB")
    container_name = os.getenv("COSMOS_CONTAINER_RAG")
    if not endpoint or not key or not db_name or not container_name:
        raise RuntimeError("Cosmos DB の環境変数が未設定です。")
    client = CosmosClient(endpoint, key)
    db = client.get_database_client(db_name)
    return db.get_container_client(container_name)

# --- 埋め込み/分割 ------------------------------------------------------

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    文字列リストをAzure OpenAIの埋め込みでベクトル化し、np.ndarray (n, d) を返す。
    """
    client = get_aoai_client()
    dep = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    resp = client.embeddings.create(model=dep, input=texts)
    vecs = [np.array(e.embedding, dtype=np.float32) for e in resp.data]
    return np.vstack(vecs)

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    """
    単純な文字数ベースのチャンク分割。PDFの段落を壊しすぎないよう重複を持たせる。
    """
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        i = j - overlap
        if i < 0:
            i = 0
        if i >= len(text):
            break
    return chunks

# --- インデクシング/検索 ------------------------------------------------

def index_pdf_to_cosmos(local_pdf_path: str, doc_id: str) -> int:
    """
    PDFをテキスト抽出→チャンク→埋め込み計算→Cosmosに保存。
    戻り値は登録したチャンク件数。
    """
    container = get_cosmos_container()
    text = extract_text(local_pdf_path) or ""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    vecs = embed_texts(chunks)
    items = []
    for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
        item = {
            "id": f"{doc_id}::chunk::{i}",
            "doc_id": doc_id,
            "chunk_id": i,
            "text": chunk,
            "embedding": vec.tolist()
        }
        items.append(item)

    for it in items:
        container.upsert_item(it)
    return len(items)

def search_similar_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """
    クエリを埋め込み化し、Cosmosから全文取得→アプリ側でコサイン類似度を計算して上位を返す。
    ※ 本番は Cosmos のベクトル検索クエリ（プレビュー）に差し替える。
    """
    container = get_cosmos_container()
    qvec = embed_texts([query])[0].reshape(1, -1)

    # 注意：本フォールバックはデータ量が小さい前提
    rows = list(container.read_all_items())
    if not rows:
        return []

    mat = np.vstack([np.array(r["embedding"], dtype=np.float32) for r in rows])
    sims = cosine_similarity(qvec, mat).ravel()
    top_idx = sims.argsort()[::-1][:top_k]
    out = []
    for idx in top_idx:
        r = rows[idx]
        out.append({
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "score": float(sims[idx])
        })
    return out
