# evidra/services/validation.py
from typing import List, Dict, Optional
import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI
from dotenv import load_dotenv
from ..utils.rag import search_similar_chunks

load_dotenv()

# ===== タイムアウト/上限（必要に応じて環境変数で調整可能） =====
TIME_BUDGET_SEC = float(os.getenv("EVIDRA_STEP2_TIME_BUDGET_SEC", "45"))   # Step2 全体の上限
PER_EDGE_TIMEOUT_SEC = float(os.getenv("EVIDRA_STEP2_PER_EDGE_TIMEOUT", "8"))
MAX_EDGES_LLMEVAL = int(os.getenv("EVIDRA_STEP2_MAX_EDGES_LLMEVAL", "100"))
MAX_CTX = int(os.getenv("EVIDRA_STEP2_MAX_CTX", "3"))

# ===== ラベル正規化（空白/全角の揺れを潰す） =====
def _norm_label(s: str) -> str:
    x = str(s).replace("\u3000", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x

def _dedup_rated(rated: List[Dict]) -> List[Dict]:
    """(source, target) ごとに effect の絶対値が最大の行を残す。"""
    uniq: Dict[tuple, Dict] = {}
    for r in rated:
        key = (_norm_label(r["source"]), _norm_label(r["target"]))
        if key not in uniq or abs(r.get("effect", 0.0)) > abs(uniq[key].get("effect", 0.0)):
            r = {**r, "source": key[0], "target": key[1]}
            uniq[key] = r
    return list(uniq.values())

# ===== Azure OpenAI クライアント =====
def get_aoai_client() -> Optional[AzureOpenAI]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01")
    if not endpoint or not key:
        return None
    try:
        return AzureOpenAI(api_key=key, api_version=api_version, azure_endpoint=endpoint)
    except Exception:
        return None

def build_system_prompt() -> str:
    return (
        "あなたは因果推論の査読者です。以下のエッジ（source→target）について、"
        "文献や一般知見に照らして『因果の有無』『因果の向き（同/違）』『正負（同/違）』を判断し、"
        "最終的に TYPE を1..5で必ず返してください。\n"
        "TYPE1=因果なし\nTYPE2=因果あり・向き同・正負同\n"
        "TYPE3=因果あり・向き同・正負違\nTYPE4=因果あり・向き違・正負同\nTYPE5=因果あり・向き違・正負違\n"
        "返答は JSON で、keys: has(boolean), dir(bool), sign(boolean), type(int), reason(短い説明)。"
    )

def edge_query_text(e: Dict) -> str:
    return f"{e['source']} が {e['target']} に与える影響（符号: {e['sign']}）についての疫学/経済学の知見"

# ===== フォールバック判定（LLM不使用） =====
def heuristic_verdict(edge: Dict) -> Dict:
    eff = float(edge.get("effect", 0.0))
    prob = float(edge.get("prob", 0.0))
    sg = edge.get("sign", "+")
    has = (abs(eff) > 1e-3) and (prob >= 0.50)
    dir_ok = True
    sign_ok = (sg == ('+' if eff >= 0 else '-'))
    if not has:
        t = 1
    else:
        if dir_ok and sign_ok: t = 2
        elif dir_ok and not sign_ok: t = 3
        elif not dir_ok and sign_ok: t = 4
        else: t = 5
    return {"has": has, "dir": dir_ok, "sign": sign_ok, "type": t, "reason": "fallback"}

# ===== 1エッジの LLM 評価 =====
def _ask_llm_for_edge(client: AzureOpenAI, dep: str, edge: Dict, contexts: List[Dict]) -> Dict:
    ctx_texts = [
        f"[doc:{c.get('doc_id','-')} chunk:{c.get('chunk_id',0)} score:{float(c.get('score',0)):.2f}]\n{c.get('text','')}"
        for c in contexts[:MAX_CTX]
    ]
    user_content = (
        f"Edge: {edge['source']} -> {edge['target']}\n"
        f"effect={edge['effect']:.3f}, prob={edge['prob']:.2f}, sign={edge['sign']}\n"
        "以下のコンテキストを参考に判定してください：\n" + "\n\n".join(ctx_texts) + "\n\n"
        "JSONのみで出力してください。"
    )
    resp = client.chat.completions.create(
        model=dep,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_content}
        ],
        timeout=PER_EDGE_TIMEOUT_SEC,
    )
    txt = (resp.choices[0].message.content or "").strip()
    data = json.loads(txt)
    if not all(k in data for k in ("has", "dir", "sign", "type")):
        raise ValueError("LLM応答に必須キーがありません")
    return data

# ===== 評価メイン（予算厳守・並列実行・フォールバック込） =====
def evaluate_edges(edges: List[Dict], rag_present: bool) -> List[Dict]:
    """
    - Step2 全体で TIME_BUDGET_SEC を厳守
    - Azure OpenAI が使えない/不調でも必ず終了
    - LLM に投げる件数は MAX_EDGES_LLMEVAL まで（残りはフォールバック）
    """
    start = time.time()
    client = get_aoai_client()
    dep = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    # ノード名の正規化を先に行う（後工程と整合）
    edges_norm = []
    for e in edges:
        edges_norm.append({
            **e,
            "source": _norm_label(e["source"]),
            "target": _norm_label(e["target"]),
        })

    # LLM不可なら全件フォールバックで即終了
    if client is None or not dep:
        rated = []
        for e in edges_norm:
            v = heuristic_verdict(e)
            rated.append({
                **e,
                "eval_has": bool(v["has"]),
                "eval_dir": bool(v["dir"]),
                "eval_sign": bool(v["sign"]),
                "type_code": int(v["type"])
                # , "citations": []
            })
        return _dedup_rated(rated)

    # LLMに回す分・フォールバック分を分割
    to_eval = edges_norm[:MAX_EDGES_LLMEVAL]
    rest = edges_norm[MAX_EDGES_LLMEVAL:]

    rated: List[Dict] = []

    # --- まずフォールバック分を即時処理（予算節約＆早く終える） ---
    for e in rest:
        v = heuristic_verdict(e)
        rated.append({
            **e,
            "eval_has": bool(v["has"]),
            "eval_dir": bool(v["dir"]),
            "eval_sign": bool(v["sign"]),
            "type_code": int(v["type"])
            # , "citations": []
        })

    # --- 残り時間の算出 ---
    def remaining_budget():
        return max(0.0, TIME_BUDGET_SEC - (time.time() - start))

    # --- 並列実行で LLM 評価 ---
    def work(e: Dict) -> Dict:
        # 予算が尽きていたら即フォールバック
        if remaining_budget() <= 0.0:
            v = heuristic_verdict(e)
            return {
                **e,
                "eval_has": bool(v["has"]),
                "eval_dir": bool(v["dir"]),
                "eval_sign": bool(v["sign"]),
                "type_code": int(v["type"])
                # , "citations": []
            }
        # RAG 取得（任意）
        contexts: List[Dict] = []
        if rag_present:
            try:
                contexts = search_similar_chunks(edge_query_text(e), top_k=MAX_CTX)
            except Exception:
                contexts = []
        # LLM 呼び出し（個別タイムアウト）
        try:
            verdict = _ask_llm_for_edge(client, dep, e, contexts)
            t = int(verdict.get("type", 2))
            cites = [{"doc_id": c.get("doc_id","-"), "page": c.get("chunk_id", 0),
                      "snippet_id": f"chunk{c.get('chunk_id',0)}"} for c in contexts[:MAX_CTX]] if contexts else []
            return {
                **e,
                "eval_has": bool(verdict.get("has", True)),
                "eval_dir": bool(verdict.get("dir", True)),
                "eval_sign": bool(verdict.get("sign", True)),
                "type_code": t
                # , "citations": cites
            }
        except Exception:
            v = heuristic_verdict(e)
            return {
                **e,
                "eval_has": bool(v["has"]),
                "eval_dir": bool(v["dir"]),
                "eval_sign": bool(v["sign"]),
                "type_code": int(v["type"])
                # , "citations": []
            }

    # スレッド数は CPU 論理コアや件数で適度に（過剰並列を避ける）
    max_workers = min(8, max(1, min(len(to_eval), os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(work, e) for e in to_eval]
        for fut in as_completed(futures, timeout=remaining_budget() or 0.001):
            try:
                rated.append(fut.result(timeout=0))  # ここでは既に完了している想定
            except Exception:
                # 何かあってもフォールバックを加える（安全第一）
                pass

    # 残ってしまった未収穫の future（予算切れ等）を安全に閉じる
    # → ここでは結果を待たず、フォールバックで埋める
    if len(rated) < len(edges_norm):
        done_pairs = {(r["source"], r["target"]) for r in rated}
        for e in edges_norm:
            pair = (e["source"], e["target"])
            if pair in done_pairs:
                continue
            v = heuristic_verdict(e)
            rated.append({
                **e,
                "eval_has": bool(v["has"]),
                "eval_dir": bool(v["dir"]),
                "eval_sign": bool(v["sign"]),
                "type_code": int(v["type"])
                # , "citations": []
            })

    return _dedup_rated(rated)

# ===== Markdown 生成 =====
def build_markdown_table(rated_edges: List[Dict]) -> str:
    lines = []
    header = "| source | target | effect | prob | sign | 因果の有無 | 向き | 正負 | TYPE |"
    sep = "|---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---|"
    lines.append(header); lines.append(sep)
    for e in rated_edges:
        has = "Yes" if e.get('eval_has', True) else "No"
        d = "同" if e.get('eval_dir', True) else "違"
        s = "同" if e.get('eval_sign', True) else "違"
        # cite = ", ".join([f"{c['doc_id']}#{c['page']}:{c['snippet_id']}" for c in e.get('citations', [])]) or "-"
        lines.append(f"| {e['source']} | {e['target']} | {e['effect']:.3f} | {e['prob']:.2f} | {e.get('sign','+')} | {has} | {d} | {s} | TYPE{int(e.get('type_code',2))} |")
    return "\n".join(lines)

# ===== tasks.py から呼ばれる互換API =====
def rate_edges(run, edges, rag_doc=None):
    rag_present = rag_doc is not None
    return evaluate_edges(edges, rag_present=rag_present)

def build_markdown_from_rated(rated_edges):
    return build_markdown_table(rated_edges)
