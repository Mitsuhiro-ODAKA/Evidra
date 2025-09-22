import networkx as nx
from typing import Dict, List, Tuple, Iterable

# DAGの階層（y）と同層内の整列（x）を決めるための単純な座標計算。

def hierarchical_layers_from_edges(edges: Iterable[Tuple[str, str]]) -> Dict[str, int]:
    """
    エッジ列から階層（レベル）を推定する。
    - ラグが明示されていれば (t-k) を下位層、(t) を最上位層にマップ
    - 明示されない場合はトポロジカル順
    """
    G = nx.DiGraph()
    for s, t in edges:
        G.add_edge(s, t)
    if G.number_of_nodes() == 0:
        return {}
    # (t-k) と (t) を手掛かりにレベルを割り当て
    layers = {}
    for node in G.nodes():
        if "(t-" in node:
            k = int(node.split("(t-")[1].split(")")[0])
            layers[node] = max(0, k)
        elif "(t)" in node:
            layers[node] = 10
        else:
            layers[node] = 5
    # 同値がある前提でOK。値が未設定のノードがあればトポ順で補完
    if any(v is None for v in layers.values()):
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            if not preds:
                layers[node] = 0
            else:
                layers[node] = max(layers[p] for p in preds) + 1
    return layers

def node_positions(edges: List[Tuple[str, str]]) -> Dict[str, Tuple[float, float]]:
    """
    階層（レベル）ごとにノードを横一列に並べ、(x,y)座標を返す。
    y = 層（小さいほど下、t層を一番上に）
    x = 同層内の並び順
    """
    layers = hierarchical_layers_from_edges(edges)
    if not layers:
        return {}
    buckets = {}
    for n, lv in layers.items():
        buckets.setdefault(lv, []).append(n)
    # 同層内の順序は名前順で安定化
    for lv in buckets:
        buckets[lv].sort()

    pos = {}
    # yは大きいほど上にしたいので、正規化して反転
    all_levels = sorted(buckets.keys())
    level_to_y = {lv: (idx + 1) * 1.0 for idx, lv in enumerate(all_levels)}  # 均等間隔
    max_y = max(level_to_y.values())
    for lv, nodes in buckets.items():
        for i, n in enumerate(nodes):
            pos[n] = (i * 1.0, max_y - level_to_y[lv])
    return pos
