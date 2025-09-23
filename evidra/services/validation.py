from typing import List, Dict, Optional
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from ..utils.rag import search_similar_chunks

load_dotenv()

import re

def _norm_label(s: str) -> str:
    x = str(s).replace("\u3000", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x

def _dedup_rated(rated):
    uniq = {}
    for r in rated:
        key = (_norm_label(r["source"]), _norm_label(r["target"]))
        if key not in uniq or abs(r.get("effect", 0.0)) > abs(uniq[key].get("effect", 0.0)):
            # 正規化を実体にも反映（Step3 へそのまま渡るため）
            r = {**r, "source": key[0], "target": key[1]}
            uniq[key] = r
    return list(uniq.values())
    
def get_aoai_client() -> Optional[AzureOpenAI]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        return None
    try:
        return AzureOpenAI(api_key=key, api_version="2024-06-01", azure_endpoint=endpoint)
    except Exception:
        return None

def build_system_prompt() -> str:
    """
    評価方針をぶれなく伝えるためのシステムプロンプト。
    """
    return (
        "あなたは因果推論の査読者です。以下のエッジ（source→target）について、"
        "文献や一般知見に照らして『因果の有無』『因果の向き（同/違）』『正負（同/違）』を判断し、"
        "最終的に TYPE を1..5で必ず返してください。\n"
        "TYPE1=因果なし\nTYPE2=因果あり・向き同・正負同\n"
        "TYPE3=因果あり・向き同・正負違\nTYPE4=因果あり・向き違・正負同\nTYPE5=因果あり・向き違・正負違\n"
        "返答は JSON で、keys: has(boolean), dir(bool), sign(bool), type(int), reason(短い説明)。"
    )

def edge_query_text(e: Dict) -> str:
    """
    RAG検索用のクエリテキストを作る（source/targetと符号を含める）。
    """
    return f"{e['source']} が {e['target']} に与える影響（符号: {e['sign']}）についての疫学/経済学の知見"

def ask_llm_with_context(edge: Dict, contexts: List[Dict]) -> Dict:
    """
    Azure OpenAI (GPT) にコンテキスト付きで評価を依頼し、構造化結果を返す。
    """
    client = get_aoai_client()
    dep = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    # クレデンシャル未設定 or デプロイ未指定ならフォールバック
    if client is None or not dep:
        return heuristic_verdict(edge)

    # コンテキストは短く絞る（上位3件程度）
    ctx_texts = [f"[doc:{c['doc_id']} chunk:{c['chunk_id']} score:{c['score']:.2f}]\n{c['text']}" for c in contexts[:3]]
    user_content = (
        f"Edge: {edge['source']} -> {edge['target']}\n"
        f"effect={edge['effect']:.3f}, prob={edge['prob']:.2f}, sign={edge['sign']}\n"
        "以下のコンテキストを参考に判定してください：\n" + "\n\n".join(ctx_texts) + "\n\n"
        "JSONのみで出力してください。"
    )

    try:
        resp = client.chat.completions.create(
            model=dep,
            temperature=0,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_content}
            ],
            timeout=25  # Step2全体120秒の中で安全側
        )
        txt = (resp.choices[0].message.content or "").strip()
        data = json.loads(txt)
        # 最低限のキーがなければ例外→フォールバック
        if not all(k in data for k in ("has","dir","sign","type")):
            raise ValueError("LLM応答に必須キーがありません")
        return data
    except Exception:
        return heuristic_verdict(edge)

def heuristic_verdict(edge: Dict) -> Dict:
    """
    LLMが使えない/失敗した場合のフォールバック。
    データ駆動（effect, prob, sign）から素朴に判定。
    """
    eff = float(edge.get("effect", 0.0))
    prob = float(edge.get("prob", 0.0))
    sg = edge.get("sign", "+")
    has = (abs(eff) > 1e-3) and (prob >= 0.50)
    dir_ok = True  # VAR-LiNGAMが与えた向きを既定で尊重
    sign_ok = (sg == ('+' if eff >= 0 else '-'))
    if not has:
        t = 1
    else:
        if dir_ok and sign_ok: t = 2
        elif dir_ok and not sign_ok: t = 3
        elif not dir_ok and sign_ok: t = 4
        else: t = 5
    return {"has": has, "dir": dir_ok, "sign": sign_ok, "type": t, "reason": "fallback"}

def evaluate_edges(edges: List[Dict], rag_present: bool) -> List[Dict]:
    """
    各エッジに対し、RAG（ある場合）で文脈を集め、Azure OpenAIで評価。
    citations には doc_id と chunk_id を入れる。
    """
    rated = []
    for e in edges:
        contexts = []
        if rag_present:
            try:
                contexts = search_similar_chunks(edge_query_text(e), top_k=5)
            except Exception:
                contexts = []
        verdict = ask_llm_with_context(e, contexts)
        t = int(verdict.get("type", 2))
        rated.append({
            **e,
            "eval_has": bool(verdict.get("has", True)),
            "eval_dir": bool(verdict.get("dir", True)),
            "eval_sign": bool(verdict.get("sign", True)),
            "type_code": t,
            "citations": [{"doc_id": c.get("doc_id","-"), "page": c.get("chunk_id", 0), "snippet_id": f"chunk{c.get('chunk_id',0)}"} for c in contexts[:3]] if contexts else []
        })
    return _dedup_rated(rated)

def build_markdown_table(rated_edges: List[Dict]) -> str:
    """
    仕様どおりのMarkdown表を生成（citations は doc_id#page:snippet_id を表示）。
    """
    lines = []
    header = "| source | target | effect | prob | sign | 因果の有無 | 向き | 正負 | TYPE | citations |"
    sep = "|---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---|"
    lines.append(header)
    lines.append(sep)
    for e in rated_edges:
        has = "Yes" if e['eval_has'] else "No"
        d = "同" if e['eval_dir'] else "違"
        s = "同" if e['eval_sign'] else "違"
        cite = ", ".join([f"{c['doc_id']}#{c['page']}:{c['snippet_id']}" for c in e.get('citations', [])]) or "-"
        lines.append(f"| {e['source']} | {e['target']} | {e['effect']:.3f} | {e['prob']:.2f} | {e['sign']} | {has} | {d} | {s} | TYPE{e['type_code']} | {cite} |")
    return "\n".join(lines)
    
# 互換ラッパー：tasks.py が期待する名前とシグネチャ
def rate_edges(run, edges, rag_doc=None):
    """
    tasks.py から呼ばれる想定のラッパー。
    - edges: [{"source","target","effect","prob","sign",...}]
    - rag_doc: None なら RAG なし、あれば RAG あり
    戻り値は evaluate_edges() と同じ構造のリストを返す。
    """
    rag_present = rag_doc is not None
    return evaluate_edges(edges, rag_present=rag_present)

# 互換：Markdown 生成関数名を tasks 側で使いやすく
def build_markdown_from_rated(rated_edges):
    return build_markdown_table(rated_edges)

