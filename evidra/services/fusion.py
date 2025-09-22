# evidra/services/fusion.py
from typing import List, Dict
import re

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

# TYPE→色/ダッシュパターン（Mermaidの linkStyle で表現）
TYPE_STYLE = {
    1: dict(color="#cccccc", width="1.2px", dash=None),                # 実線（薄）
    2: dict(color="#00338D", width="2.4px", dash=None),                # 実線
    3: dict(color="#7213EA", width="2.0px", dash="6 4"),              # 破線
    4: dict(color="#00B8F5", width="2.0px", dash="2 3"),              # 点線相当（短い破線）
    5: dict(color="#FD349C", width="2.0px", dash="10 3 2 3"),         # ダッシュドット相当
}

def build_mermaid_fusion(rated_edges: List[Dict]) -> str:
    """
    Step3 の融合グラフ：Step1 と同じ flowchart 構文に統一し、
    エッジの見た目だけ linkStyle で TYPE 別に変更する（凡例なし）。
    """
    if not rated_edges:
        return "flowchart TD\n    N_empty[\"表示可能なエッジがありません\"]"

    # ノード集合
    nodes = sorted({e["source"] for e in rated_edges} | {e["target"] for e in rated_edges})

    # 安全なIDに置換（重複は連番で回避）
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

    # エッジ本体（**ここでの並び順が linkStyle の index になる**）
    link_types = []  # linkStyle 用に TYPE を記録
    for e in rated_edges:
        s_id = id_map[e["source"]]
        t_id = id_map[e["target"]]
        lines.append(f'    {s_id} --> {t_id}')
        link_types.append(int(e.get("type_code", 1)))

    # linkStyle で TYPE 別スタイルを適用
    # 例: linkStyle 0 stroke:#00338D,stroke-width:2.4px,stroke-dasharray:6 4;
    for idx, tcode in enumerate(link_types):
        style = TYPE_STYLE.get(tcode, TYPE_STYLE[1])
        parts = [f"stroke:{style['color']}", f"stroke-width:{style['width']}"]
        if style["dash"]:
            parts.append(f"stroke-dasharray:{style['dash']}")
        # Mermaid v11 はカンマ区切りで OK（末尾セミコロンは付けない）
        lines.append(f"    linkStyle {idx} " + ",".join(parts))

    return "\n".join(lines)
