from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from django.conf import settings
from urllib.parse import urljoin

# TYPEごとの線種（Plotlyのdash指定）
TYPE_DASH = {
    1: "solid",    # gray solid
    2: "solid",    # solid
    3: "dash",     # dashed
    4: "dot",      # dotted
    5: "dashdot",  # wave 代替
}

TYPE_COLOR = {
    1: "#cccccc",
    2: "#00338D",
    3: "#7213EA",
    4: "#00B8F5",
    5: "#FD349C",
}

def export_graph_html(rated_edges: List[Dict], positions: Dict[str, Tuple[float, float]], out_name: str) -> str:
    """
    階層座標を用いてPlotlyでノード・エッジを描画し、HTMLで保存する。
    - rated_edges: {source,target,type_code,...}の辞書配列
    - positions  : {node: (x,y)} の座標辞書
    """
    # ノード一覧
    nodes = sorted({e["source"] for e in rated_edges} | {e["target"] for e in rated_edges})
    fig = go.Figure()

    safe_positions = positions or {}
    if not nodes or not safe_positions:
        # 何も描けない場合はプレースホルダを出す（Step3を失敗させない）
        fig.add_annotation(text="表示可能なエッジがありません（Step1/2の結果をご確認ください）",
                           x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        xs, ys = [], []
    else:
        xs = [safe_positions.get(n, (nodes.index(n), 0))[0] for n in nodes]
        ys = [safe_positions.get(n, (nodes.index(n), 0))[1] for n in nodes]
        # ノード（丸＋ラベル）— hover と text で確実に見えるようにする
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            textfont=dict(size=12),
            hoverinfo="text",
            name="ノード",
            showlegend=True,
            marker=dict(size=12, line=dict(width=1, color="#333"))
        ))

    fig = go.Figure()

    # エッジは TYPE ごとに 1 トレースへまとめ、凡例を1回だけ出す
    if nodes and safe_positions:
        by_type: Dict[int, List[Tuple[float,float,float,float,str]]] = {}
        for e in rated_edges:
            s, t = e["source"], e["target"]
            if s not in safe_positions or t not in safe_positions:
                continue
            x0, y0 = safe_positions[s]
            x1, y1 = safe_positions[t]
            tcode = int(e.get("type_code", 1))
            by_type.setdefault(tcode, []).append((x0, y0, x1, y1, f"{s}→{t} | TYPE{tcode} | effect={e['effect']:.3f} prob={e['prob']:.2f}"))

        for tcode, segs in sorted(by_type.items()):
            # 複数セグメントを1トレースにまとめる（None 区切り）
            xs, ys, texts = [], [], []
            for x0,y0,x1,y1,txt in segs:
                xs.extend([x0, x1, None])
                ys.extend([y0, y1, None])
                texts.extend([txt, txt, None])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=TYPE_COLOR.get(tcode, "#999999"),
                          width=2,
                          dash=TYPE_DASH.get(tcode, "solid")),
                name=f"TYPE{tcode}",
                hoverinfo="text",
                text=texts,
                showlegend=True
            ))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=24, b=10),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    out_dir = Path(settings.MEDIA_ROOT) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    fig.write_html(str(out_path), include_plotlyjs='cdn')
    # ブラウザで開けるURLを返す（例：/media/plots/run_123_graph.html）
    public_url = urljoin(settings.MEDIA_URL, f"plots/{out_name}")
    return public_url
