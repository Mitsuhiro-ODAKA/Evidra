# evidra/services/fusion.py
from typing import List, Dict
import re

# ---------- 共通ユーティリティ ----------

def _norm_label(s: str) -> str:
    """ノード名の正規化（前後/内部空白のゆれ、全角空白、改行などを吸収）"""
    x = str(s).replace("\u3000", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x

def _safe_id(label: str) -> str:
    """MermaidノードIDとして安全なID（先頭は必ず英字）。"""
    sid = re.sub(r'[^0-9A-Za-z_]', '_', label)
    sid = re.sub(r'_+', '_', sid).strip('_')
    if not sid or not sid[0].isalpha():
        sid = f"N_{sid or 'X'}"
    return sid

def _esc(text: str) -> str:
    """Mermaidのノード/エッジラベルに入れる文字の最小エスケープ。"""
    if text is None:
        return ""
    return str(text).replace('"', '&quot;').replace('\n', ' ').replace('\r', ' ')

# 線種（TYPE → dash パターン）※色は sign で決める
TYPE_DASH = {
    2: None,                 # 実線
    3: "6 6",                # 破線
    4: "2 8",                # 点線相当
    5: "8 5 2 5",            # ダッシュドット相当
    1: None,                 # TYPE1 は実線だが色は薄めグレーに落とす（任意）
}

def _dedup_rated(rated_edges: List[Dict]) -> List[Dict]:
    """(source,target) 正規化キーで一意化。代表は |effect| が大きい方（好みで prob に変更可）"""
    uniq = {}
    for r in rated_edges:
        s = _norm_label(r["source"])
        t = _norm_label(r["target"])
        key = (s, t)
        cur = uniq.get(key)
        if (cur is None) or (abs(r.get("effect", 0.0)) > abs(cur.get("effect", 0.0))):
            # 正規化を実体にも反映しておく（下流でそのまま使えるように）
            rr = dict(r)
            rr["source"] = s
            rr["target"] = t
            uniq[key] = rr
    # 安定した順序で返す（source,target の辞書順）
    return [uniq[k] for k in sorted(uniq.keys())]

# ---------- 公開関数 ----------

def build_mermaid_fusion(rated_edges: List[Dict]) -> str:
    """
    Step3 の融合グラフ（Mermaid flowchart TD）:
      - 入力は Step2 の rated_edges（source/target/effect/prob/sign/type_code）
      - (source,target) で重複除去
      - 色は sign で決定（+ = 赤, - = 青, TYPE1 は薄グレー）
      - 線種は TYPE（2=solid, 3=dashed, 4=dotted, 5=dashdot）
    """
    if not rated_edges:
        return 'flowchart TD\n    N_empty["表示可能なエッジがありません"]'

    edges = _dedup_rated(rated_edges)

    # ノード集合（正規化済み）
    nodes = sorted({e["source"] for e in edges} | {e["target"] for e in edges})

    # 安全なIDに置換（衝突時は連番で回避）
    id_map, used = {}, set()
    for label in nodes:
        base = _safe_id(label)
        sid, k = base, 1
        while sid in used:
            k += 1
            sid = f"{base}_{k}"
        id_map[label] = sid
        used.add(sid)

    lines = ["flowchart TD"]

    # ノード
    for label, sid in id_map.items():
        lines.append(f'    {sid}["{_esc(label)}"]')

    # エッジ（ここでの並び順が linkStyle の index）
    link_styles = []
    for idx, e in enumerate(edges):
        s_id = id_map[e["source"]]
        t_id = id_map[e["target"]]
        lines.append(f"    {s_id} --> {t_id}")

        # 線種：TYPE → dash パターン（既存どおり）
        tcode = int(e.get("type_code", 2))
        dash = TYPE_DASH.get(tcode)

        # ---- 色決定：TYPE2 だけ符号色、他 TYPE は薄いグレー ----
        # sign / sign_symbol / effect の順に符号を決め、TYPE2 の時だけ使う
        color = "#b6b6b6"  # 既定：薄いグレー（TYPE2 以外）
        if tcode == 2:
            sym = str(e.get("sign", "")).strip()
            if sym not in {"+", "-"}:
                sym = str(e.get("sign_symbol", "")).strip()
            if sym not in {"+", "-"}:
                try:
                    eff = float(e.get("effect", 0.0))
                    sym = "+" if eff >= 0 else "-"
                except Exception:
                    sym = "+"
            color = "#e53935" if sym == "+" else "#1e3a8a"  # 正=赤 / 負=青

        parts = [f"stroke:{color}", "stroke-width:2px"]
        if dash:
            parts.append(f"stroke-dasharray:{dash}")
        lines.append(f"    linkStyle {idx} " + ",".join(parts))

    return "\n".join(lines)
