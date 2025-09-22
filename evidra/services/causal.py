import pandas as pd
import numpy as np
import networkx as nx
import re
from math import ceil, sqrt
from typing import List, Dict, Tuple
from lingam.var_lingam import VARLiNGAM

# --- ヘルパ：Moving Block Bootstrap (MBB) ------------------------------

def moving_block_bootstrap(df: pd.DataFrame, block_len: int, n_boot: int, rng: np.random.Generator) -> List[pd.DataFrame]:
    """
    時系列の依存構造を大きく壊さないよう、固定長ブロックを並べ替えて再標本化する。
    - 各ブートで、全長をカバーするまでブロックをランダム接続する。
    """
    T = len(df)
    idx = np.arange(T)
    # T が短い場合のガード：最低1ブロックは作る
    block_len = max(2, min(block_len, T))
    blocks = [idx[i:i+block_len] for i in range(0, T - block_len + 1)] or [idx]
    
    boots = []
    for _ in range(n_boot):
        seq = []
        while len(seq) < T:
            b = blocks[rng.integers(0, len(blocks))]
            seq.extend(b.tolist())
        seq = seq[:T]
        boots.append(df.iloc[seq].reset_index(drop=True))
    return boots

# --- メイン：VAR-LiNGAM 実行 -------------------------------------------

def run_var_lingam(df: pd.DataFrame, lag: int, boot: int, seed: int) -> Tuple[List[Dict], str]:
    """
    VAR-LiNGAM を用いて因果構造を推定し、エッジリストとMermaidコードを返す。
    - effect: 係数推定値（標準化はしない）
    - prob  : MBB (ceil(sqrt(T))) による出現頻度
    """
    rng = np.random.default_rng(seed)
    cols = list(df.columns)
    X = df.values

    # lingamのVARLiNGAMは内部で標準化するが、入力は float & NaN 無しが前提
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("入力にNaN/Infが含まれています。前処理（欠損補間など）を確認してください。")
    if X.shape[0] <= max(10, lag*2):
        raise ValueError(f"時系列長が短すぎます（T={X.shape[0]}）。ラグ={lag}に対して十分な長さが必要です。")
    model = VARLiNGAM(lags=lag, criterion=None, random_state=seed)
    model.fit(X)

    # adjacency_matrices_: 形状は (lags+1, n_features, n_features)
    # 0番目が同時点（t）間のB、以降は遅れ辺のA(1..lag)
    A_mats = model.adjacency_matrices_
    n = len(cols)

    # エッジ集合（方向/符号/効果）を抽出
    edges = []
    for k in range(1, lag+1):               # 遅れ辺: X_{t-k} -> X_t
        A = A_mats[k]
        for j in range(n):                  # target (t)
            for i in range(n):              # source (t-k)
                coef = A[j, i]
                if np.abs(coef) < 1e-6:
                    continue
                sign = '+' if coef >= 0 else '-'
                edges.append({
                    "source": f"{cols[i]}(t-{k})",   # 表示上、ラグを明示
                    "target": f"{cols[j]}(t)",
                    "effect": float(coef),
                    "prob": 1.0,                     # 後でMBBで上書き
                    "sign": sign
                })

    # 同時点の構造（B行列）：X_t の間のDAG（LiNGAM）。B[j,i] は i->j の重み
    B = A_mats[0]
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            coef = B[j, i]
            if np.abs(coef) < 1e-6:
                continue
            sign = '+' if coef >= 0 else '-'
            edges.append({
                "source": f"{cols[i]}(t)",
                "target": f"{cols[j]}(t)",
                "effect": float(coef),
                "prob": 1.0,
                "sign": sign
            })

    # --- MBB による出現頻度推定 ---------------------------------------
    T = len(df)
    block_len = int(ceil(sqrt(T)))          # 仕様：ceil(sqrt(T))
    boots = moving_block_bootstrap(df, block_len, boot, rng)

    # 出現判定は「同じ向き・同じ両端の辺が非ゼロ」かつ「符号が一致」。
    def edge_key(e):  # ソース/ターゲット/ラグ表示をキーにする
        return (e["source"], e["target"])

    edge_map = {edge_key(e): e for e in edges}
    counts = {edge_key(e): 0 for e in edges}

    for df_b in boots:
        model_b = VARLiNGAM(lags=lag, criterion=None, random_state=seed)
        model_b.fit(df_b.values)
        A_b = model_b.adjacency_matrices_
        # 遅れ辺
        for k in range(1, lag+1):
            A = A_b[k]
            for j in range(n):
                for i in range(n):
                    coef = A[j, i]
                    if np.abs(coef) < 1e-6:
                        continue
                    s = '+' if coef >= 0 else '-'
                    key = (f"{cols[i]}(t-{k})", f"{cols[j]}(t)")
                    if key in counts and edge_map[key]["sign"] == s:
                        counts[key] += 1
        # 同時点
        B_b = A_b[0]
        for j in range(n):
            for i in range(n):
                if i == j:
                    continue
                coef = B_b[j, i]
                if np.abs(coef) < 1e-6:
                    continue
                s = '+' if coef >= 0 else '-'
                key = (f"{cols[i]}(t)", f"{cols[j]}(t)")
                if key in counts and edge_map[key]["sign"] == s:
                    counts[key] += 1

    # probを出現頻度/bootに置換
    for e in edges:
        e["prob"] = float(counts[edge_key(e)] / max(1, boot))

    # --- Mermaid（階層レイアウト） ------------------------------------
    # 表示上は (t-k) 層 → (t) 層の階層構造にする（同時点内の辺は同層）
    G = nx.DiGraph()
    nodes = set()
    for e in edges:
        G.add_edge(e["source"], e["target"])
        nodes.add(e["source"]); nodes.add(e["target"])

    # ここでは簡易層割り：末尾が"(t-k)"のkで層を決め、"(t)"は最大層
    layers = {}
    for nname in nodes:
        if "(t-" in nname:
            k = int(nname.split("(t-")[1].split(")")[0])
            layers[nname] = max(0, k) * 10       # ラグ差を大きめに離す
        else:
            layers[nname] = 100

    # Mermaidコード生成（IDは英数と_のみ。重複は連番で回避）
    def safe_id(label: str) -> str:
        sid = re.sub(r'[^0-9A-Za-z_]', '_', label)
        sid = re.sub(r'_+', '_', sid).strip('_')
        if not sid or not sid[0].isalpha():
            sid = f"N_{sid or 'X'}"
        return sid
    def esc_label(text: str) -> str:
        return (str(text).replace('"','&quot;').replace('\n',' ').replace('\r',' ')) if text is not None else ""
    id_map = {}
    used = set()
    for label in nodes:
        base = safe_id(label)
        sid = base
        k = 1
        while sid in used:
            k += 1
            sid = f"{base}_{k}"
        id_map[label] = sid
        used.add(sid)

    lines = ["flowchart TD"]
    for label, sid in id_map.items():
        lines.append(f'    {sid}["{esc_label(label)}"]')
    for e in edges:
        src = id_map[e["source"]]
        tgt = id_map[e["target"]]
        lines.append(f'    {src} --> {tgt}')
    mermaid_code = "\n".join(lines)

    return edges, mermaid_code
