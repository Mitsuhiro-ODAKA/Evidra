# evidra/tasks.py
from __future__ import annotations

import math
import json
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import threading
import re

import numpy as np
import pandas as pd
from django.utils import timezone

from .models import Run, Dataset, RagDoc, Edge, Artifact


def _norm_label(s: str) -> str:
    """ノード名の正規化（前後空白/改行/全角空白の揺れを潰す）"""
    x = str(s).replace("\u3000", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x
    
# 可能な限り、既存サービスを利用（存在しない場合は後述のフォールバックに切替）
try:
    # Step3 の TYPE 別スタイル付与済み Mermaid を生成
    from .services.fusion import build_mermaid_fusion as fusion_mermaid
except Exception:  # pragma: no cover
    fusion_mermaid = None

try:
    # Plotly HTML を /media/plots/...html で保存し、公開URLを返す
    from .utils.plotly_export import export_graph_html
except Exception:  # pragma: no cover
    export_graph_html = None

try:
    # TYPE 付与などの評価ロジック（RAG/LLM あり版）
    from .services.validation import rate_edges as rate_edges_llm
except Exception:  # pragma: no cover
    rate_edges_llm = None

# LiNGAM（VAR-LiNGAM）の実装ライブラリ
try:
    from lingam import VARLiNGAM
except Exception as _e:  # pragma: no cover
    VARLiNGAM = None

# ============================================================
# ステータス管理（未実行/処理中/完了/失敗）を一貫して更新
# ============================================================

def _set_status(
    run: Run,
    *,
    step: int,
    pct: int,
    label: str,
    stage_statuses: Optional[dict] = None,
    overall: Optional[str] = None,
) -> None:
    """Run.status を安全に更新する共通関数。"""
    st = run.status or {}
    st["step"] = step
    st["pct"] = int(max(0, min(100, pct)))
    st["label"] = label
    if stage_statuses:
        ss = st.get("stage_statuses", {})
        ss.update(stage_statuses)
        st["stage_statuses"] = ss
    st.setdefault(
        "stage_statuses",
        {"Step1": "未実行", "Step2": "未実行", "Step3": "未実行"},
    )
    if overall:
        st["overall"] = overall
    else:
        # ラベルに応じた overall 既定
        if label == "処理中":
            st.setdefault("overall", "Running")
        elif label == "完了":
            st.setdefault("overall", "Succeeded")
        elif label == "失敗":
            st.setdefault("overall", "Failed")
        else:
            st.setdefault("overall", "Pending")
    run.status = st
    run.save(update_fields=["status"])


# ============================================================
# ランチャ：UI から呼ばれた直後に Running へ遷移
# ============================================================

def launch_run(run_id: int) -> None:
    """
    Run をバックグラウンドで起動し、すぐに呼び元へ戻る。
    フロントの /status ポーリングにより「処理中」→「完了/失敗」を逐次反映できる。
    """
    run = Run.objects.get(id=run_id)
    _set_status(
        run,
        step=1,
        pct=1,
        label="処理中",
        stage_statuses={"Step1": "処理中", "Step2": "未実行", "Step3": "未実行"},
        overall="Running",
    )
    t = threading.Thread(target=run_pipeline_async, args=(run.id,), daemon=True)
    t.start()


# ============================================================
# データ読み込み・前処理ユーティリティ
# ============================================================

def _load_observation_frame(dataset: Dataset) -> pd.DataFrame:
    """CSV/XLSX を読み込み、先頭行=列名を前提に DataFrame へ変換する。"""
    path = dataset.file_path
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)  # infer_datetime_format は使わない（将来非推奨）
    elif path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError("CSV または XLSX をサポートします。")
    if df.empty:
        raise ValueError("観測データが空です。")
    return df


def _choose_time_column(df: pd.DataFrame) -> Optional[str]:
    """日付列を推定（'日付'優先、無ければ datetime 変換で候補を探す）。"""
    # まず日本語の「日付」列があればそれを採用
    if "日付" in df.columns:
        try:
            pd.to_datetime(df["日付"], errors="raise")  # 変換可能か簡易チェック
            return "日付"
        except Exception:
            pass
    # 先頭数列から datetime に変換可能な列を探索
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() > 0.95:
            # 95%以上が日時として解釈できれば採用
            return c
    return None


def _coerce_numeric(df: pd.DataFrame, ignore_cols: List[str]) -> pd.DataFrame:
    """指定列以外を可能な限り数値化し、欠損は線形補間→前方埋めで補完。"""
    out = df.copy()
    for c in out.columns:
        if c in ignore_cols:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # 欠損の補完（線形→前方）
    out = out.interpolate(limit_direction="both")
    out = out.fillna(method="ffill").fillna(method="bfill")
    return out


def _standardize_if_needed(X: np.ndarray, enable: bool) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """既定ONの標準化（出力は UI に明記せず、内部で用いるのみ）。"""
    if not enable:
        return X, None, None
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd


# ============================================================
# VAR-LiNGAM 推定 + MBB ブート頻度（prob）
# ============================================================

@dataclass
class EdgeEst:
    source: str     # 例: "接種率_パーセント(t-1)"
    target: str     # 例: "インフルエンザ患者数(t)"
    effect: float
    prob: float
    sign: str
    lag: int        # 1..p（p は VAR のラグ数）

def _var_lingam_edges(
    df: pd.DataFrame,
    var_cols: List[str],
    lag: int,
    boot: int,
    seed: int,
    standardize: bool = True,
    thr: float = 1e-8,          # ← 追加：外部から指定可（既定は緩め）
) -> Tuple[List[EdgeEst], List[str]]:

    if VARLiNGAM is None:
        raise RuntimeError("lingam パッケージが見つかりません（pip install lingam）")

    X = df[var_cols].to_numpy(dtype=float)
    n, d = X.shape
    if n < lag + 5:
        raise ValueError("系列長が短すぎます（lag に対して十分な行数が必要）")

    Xz, _, _ = _standardize_if_needed(X, enable=standardize)

    base_model = VARLiNGAM(lags=lag, random_state=seed)
    base_model.fit(Xz)
    mats = np.array(base_model.adjacency_matrices_)  # (L, d, d)

    rng = np.random.default_rng(seed)
    B = max(1, int(boot))
    L = int(math.ceil(math.sqrt(n)))  # MBB ブロック長

    def mbb_indices(n: int, L: int) -> np.ndarray:
        blocks = []
        k = int(math.ceil(n / L))
        for _ in range(k):
            s = rng.integers(0, n - L + 1)
            blocks.append(np.arange(s, s + L))
        idx = np.concatenate(blocks)[:n]
        return idx

    present = np.zeros((lag, d, d), dtype=int)
    for b in range(B):
        idx = mbb_indices(n, L)
        Xb = Xz[idx]
        try:
            m = VARLiNGAM(lags=lag, random_state=seed + b + 1)
            m.fit(Xb)
            mats_b = np.array(m.adjacency_matrices_)
            mask_b = (np.abs(mats_b) >= thr).astype(int)   # ← 閾値を使用
            present += mask_b
        except Exception:
            continue

    # ノード集合（t と t-k）
    all_nodes: List[str] = []
    all_nodes.extend([f"{v}(t)" for v in var_cols])
    for k in range(1, lag + 1):
        all_nodes.extend([f"{v}(t-{k})" for v in var_cols])

    #  (lag, i, j) をキーに一意化（重複エッジ根絶）
    seen: set[Tuple[int, int, int]] = set()
    edges: List[EdgeEst] = []
    for k in range(lag):                 # 0..lag-1 → 表示は (t-(k+1))
        M = mats[k]
        for j in range(d):               # target: 変数 j at t
            for i in range(d):           # source: 変数 i at t-(k+1)
                if i == j and k == 0:
                    continue
                eff = float(M[i, j])
                if abs(eff) < thr:
                    continue
                key = (k, i, j)
                if key in seen:
                    continue             # ← 二重登録を防止
                seen.add(key)

                freq = float(present[k, i, j] / max(1, B))
                sign = "+" if eff >= 0 else "-"
                src = _norm_label(f"{var_cols[i]}(t-{k+1})")
                tgt = _norm_label(f"{var_cols[j]}(t)")
                edges.append(EdgeEst(source=src, target=tgt, effect=eff, prob=freq, sign=sign, lag=k+1))
    return edges, all_nodes


# ============================================================
# Mermaid（Step1 の最小構文）
# ============================================================

def _build_mermaid_step1(nodes: List[str], edges: List[EdgeEst]) -> str:
    """Mermaid v11 向けの素朴な flowchart。IDは英字開始に正規化。"""
    import re

    uniq = {}
    for e in edges:
        key = (_norm_label(e.source), _norm_label(e.target))
        # 代表値は effect が大きいものを採用（好みで max prob でもOK）
        if key not in uniq or abs(e.effect) > abs(uniq[key].effect):
            uniq[key] = e
    edges = list(uniq.values())
    
    def sid(label: str) -> str:
        s = re.sub(r"[^0-9A-Za-z_]", "_", label).strip("_")
        if not s or not s[0].isalpha():
            s = f"N_{s or 'X'}"
        return s

    uniq_nodes = sorted({_norm_label(n) for n in nodes})
    ids = {label: sid(label) for label in uniq_nodes}
    lines = ["flowchart TD"]
    for label in uniq_nodes:
        safe_label = label.replace('"', "&quot;")
        lines.append(f'    {ids[label]}["{safe_label}"]')
    # エッジは装飾なし（Step1は最小構文）
    for e in edges:
        lines.append(f"    {ids[e.source]} --> {ids[e.target]}")
    return "\n".join(lines)

def _build_mermaid_step3_colored(nodes: List[str], rated_edges: List[dict]) -> str:
    """
    Mermaid v11 flowchart:
    - エッジを符号で色分け（+ = 赤, - = 青）
    - TYPE2=solid, TYPE3=dashed, TYPE4=dotted, TYPE5=dashdot
      （stroke-dasharray を linkStyle で設定）
    """
    import re

    uniq = {}
    for r in rated_edges:
        key = (_norm_label(r["source"]), _norm_label(r["target"]))
        if key not in uniq or abs(r.get("effect", 0.0)) > abs(uniq[key].get("effect", 0.0)):
            uniq[key] = r
    rated_edges = list(uniq.values())

    def sid(label: str) -> str:
        s = re.sub(r"[^0-9A-Za-z_]", "_", label).strip("_")
        if not s or not s[0].isalpha():
            s = f"N_{s or 'X'}"
        return s

    ids = {label: sid(_norm_label(label)) for label in nodes}
    lines = ["flowchart TD"]

    # ノード
    for label in nodes:
        safe = label.replace('"', "&quot;")
        lines.append(f'    {ids[label]}["{safe}"]')

    # エッジ本体（Mermaid はエッジ定義順に 0,1,2,.. が付く）
    linkstyles = []
    for idx, r in enumerate(rated_edges):
        s = ids.get(_norm_label(r["source"]))
        t = ids.get(_norm_label(r["target"]))
        if not s or not t or s == t:
            continue
        lines.append(f"    {s} --> {t}")

        # 色：+ = 赤, - = 青
        color = "#e53935" if str(r.get("sign", "+")).startswith("+") else "#1e3a8a"
        # TYPE ごとの線種
        tp = int(r.get("type_code", 2))
        if tp == 2:       # solid
            dash = ""
        elif tp == 3:     # dashed
            dash = "stroke-dasharray: 6 6;"
        elif tp == 4:     # dotted
            dash = "stroke-dasharray: 2 8;"
        else:             # TYPE5 → dashdot っぽく
            dash = "stroke-dasharray: 8 5 2 5;"

        # linkStyle n に stroke, stroke-width, dash を設定
        linkstyles.append(
            f"    linkStyle {idx} stroke:{color},stroke-width:2px;{dash}"
        )

    lines.extend(linkstyles)
    return "\n".join(lines)
    
# ============================================================
# Step2 評価（RAG無し可）→ Markdown テーブル
# ============================================================

def _rate_edges_simple(edges: List[EdgeEst]) -> List[dict]:
    """
    フォールバックの評価器：
    - RAG/LLM が使えない場合、構文上の整合だけで TYPE2 扱いに寄せる簡易版。
    - citations はダミー（'-'）。UI 側のクリック展開は実接続時に有効化。
    """
    rated = []
    for e in edges:
        # 因果はある（Step1で抽出済み）、向きは同じ、正負は推定 sign に従う → TYPE2 とする
        rated.append({
            "source": e.source,
            "target": e.target,
            "effect": e.effect,
            "prob": e.prob,
            "has_causality": "Yes",
            "direction": "Same",
            "sign": e.sign,
            "type_code": 2,             # 既定で TYPE2
            "citations": "-",           # RAG 実装時は "DOC#pX, DOC#pY" のように埋める
        })
    return rated


def _rated_to_markdown(rated: List[dict]) -> str:
    """評価結果を Markdown テーブルへ整形（UI 側で HTML <table> へレンダリング）。"""
    headers = [
        "source", "target", "effect", "prob",
        "因果の有無", "因果の向き", "因果の正負", "正負(推定)", "TYPE", "citations"
    ]
    sep = "|".join(["---"] * len(headers))
    lines = ["|" + "|".join(headers) + "|", "|" + sep + "|"]
    for r in rated:
        row = [
            str(r["source"]),
            str(r["target"]),
            f"{r['effect']:.4f}",
            f"{r['prob']:.2f}",
            str(r.get("has_causality", "")),
            str(r.get("direction", "")),
            str(r.get("sign", "")),              # 「同/違」（評価）
            str(r.get("sign_symbol", "")),       # 「+/-」（推定符号）
            f"TYPE{int(r.get('type_code', 1))}",
            str(r.get("citations", "-")),
        ]
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)



# ============================================================
# Step3 レイアウト（階層座標）フォールバック
# ============================================================

def _simple_hier_positions(nodes: List[str], edges: List[EdgeEst]) -> Dict[str, Tuple[float, float]]:
    """
    簡易な階層レイアウト：
    - 入次数の小さいノードから層を作り、横一列に並べる。
    - 依存関係が循環していても落ちないように適当に広げる。
    """
    indeg = {n: 0 for n in nodes}
    for e in edges:
        indeg[e.target] = indeg.get(e.target, 0) + 1
    # 入次数=0 → 層0、残りは入次数順に層割り（ざっくり）
    order = sorted(nodes, key=lambda n: (indeg.get(n, 0), n))
    layer_of = {n: 0 for n in nodes}
    for n in order:
        parents = [e.source for e in edges if e.target == n]
        if parents:
            layer_of[n] = max(layer_of[p] for p in parents) + 1
        else:
            layer_of[n] = 0
    # 層ごとに y を低く、x を等間隔で配置
    layers = {}
    for n, L in layer_of.items():
        layers.setdefault(L, []).append(n)
    positions = {}
    for L, group in sorted(layers.items()):
        k = len(group)
        xs = np.linspace(0.0, 1.0, max(2, k))
        for i, n in enumerate(group):
            positions[n] = (float(xs[i]), float(1.0 - 0.2 * L))
    return positions


# ============================================================
# パイプライン本体（Step1→Step2→Step3）
# ============================================================

def run_pipeline_async(run_id: int) -> None:
    run = Run.objects.select_related("dataset", "rag_doc").get(id=run_id)
    art, _ = Artifact.objects.get_or_create(run=run)

    # 2分以内に終わる設計（ブート回数やデータ量の上限により担保）
    try:
        # ------------------ Step1: 因果探索 ------------------
        _set_status(run, step=1, pct=10, label="処理中", stage_statuses={"Step1": "処理中"})

        df = _load_observation_frame(run.dataset)

        # 時系列列の検出とソート（あれば）
        tcol = _choose_time_column(df)
        if tcol:
            # 厳密な datetime 変換（infer_datetime_format は使わない）
            dt = pd.to_datetime(df[tcol], errors="coerce")
            df = df.loc[dt.notna()].copy()
            df.sort_values(by=tcol, inplace=True)
            df.reset_index(drop=True, inplace=True)

        # 解析対象の変数列（日時列は除外）
        var_cols = [c for c in df.columns if c != tcol]

        # 数値化・補完
        df_num = _coerce_numeric(df[var_cols], ignore_cols=[])

        # パラメータ（既定値）
        params = run.params or {}
        lag = int(params.get("lag", 2))
        boot = int(params.get("boot", 100))
        seed = int(params.get("seed", 42))
        preprocessing = params.get("preprocessing", {})
        standardize = bool(preprocessing.get("standardize", True))  # 既定 ON
        
        edge_thr = float(params.get("edge_threshold", 1e-10))

        # VAR-LiNGAM 推定 + MBB prob
        edges, all_nodes = _var_lingam_edges(
            df=df_num,
            var_cols=var_cols,
            lag=lag,
            boot=boot,
            seed=seed,
            standardize=standardize,
            thr=edge_thr, 
        )

        # DBへ保存（過去のエッジをクリア）
        Edge.objects.filter(run=run).delete()
        edge_objs = []
        for e in edges:
            edge_objs.append(
                Edge(
                    run=run,
                    source=_norm_label(e.source),
                    target=_norm_label(e.target),
                    effect=e.effect,
                    prob=e.prob,
                    sign=e.sign,
                    type_code=None,  # Step2 で付与
                )
            )
        if edge_objs:
            Edge.objects.bulk_create(edge_objs)

        # Mermaid（Step1：最小構文）
        nodes_step1 = sorted({_norm_label(e.source) for e in edges} | {_norm_label(e.target) for e in edges})
        # mermaid1 = _build_mermaid_step1(nodes=nodes_step1, edges=edges)
        # art.mermaid_step1 = mermaid1
        # art.save(update_fields=["mermaid_step1"])
        art.mermaid_step1 = ""
        art.save(update_fields=["mermaid_step1"])

        _set_status(run, step=1, pct=33, label="処理中", stage_statuses={"Step1": "完了"})

        # ------------------ Step2: 妥当性評価 ------------------
        _set_status(run, step=2, pct=34, label="処理中", stage_statuses={"Step2": "処理中"})

        # 既存の LLM/RAG 評価器があれば使用。無ければ簡易評価にフォールバック。
        # ★ EdgeEst → dict へ明示変換（validation 側は dict を想定）
        edges_dicts: List[dict] = [
            {
                "source": _norm_label(e.source),
                "target": _norm_label(e.target),
                "effect": float(e.effect),
                "prob": float(e.prob),
                "sign": e.sign,
                "lag": int(e.lag),
            }
            for e in edges
        ]
        rated: List[dict]
        if rate_edges_llm is not None:
            try:
                rag: Optional[RagDoc] = run.rag_doc
                rated = rate_edges_llm(run, edges_dicts, rag_doc=rag)
            except Exception:
                rated = _rate_edges_simple(edges)
        else:
            rated = _rate_edges_simple(edges)

        # === 表示用フィールドへ正規化（評価器のキー揺れに対応） ===
        normed = []
        for r in rated:
            # eval_* → 表示用へマッピング
            has_bool = bool(r.get("eval_has", r.get("has", True)))
            dir_bool = bool(r.get("eval_dir", r.get("dir", True)))
            sign_bool = bool(r.get("eval_sign", r.get("sign_eval", True)))  # 「正負の同/違」評価
            # 推定符号（+/-）：Step1 推定の符号を保持（色分け用にも利用）
            sign_symbol = str(r.get("sign_symbol", r.get("sign", "+")))
            normed.append({
                "source": _norm_label(r["source"]),
                "target": _norm_label(r["target"]),
                "effect": float(r["effect"]),
                "prob": float(r["prob"]),
                "type_code": int(r.get("type_code", 2)),
                # 表示用（日本語列）
                "has_causality": "Yes" if has_bool else "No",
                "direction": "同" if dir_bool else "違",
                "sign": "同" if sign_bool else "違",          # ← 「因果の正負（同/違）」評価
                "sign_symbol": "+" if str(sign_symbol).startswith("+") else "-",  # ← 正/負の符号
                "citations": r.get("citations", "-"),
            })
        rated = normed

        # TYPE を Edge レコードに反映
        typemap: Dict[Tuple[str, str], int] = {}
        for r in rated:
            typemap[(r["source"], r["target"])] = int(r.get("type_code", 1))
        edge_qs = list(Edge.objects.filter(run=run))  # ← 実体化
        for eo in edge_qs:
            key = (eo.source, eo.target)
            if key in typemap:
                eo.type_code = typemap[key]
        if edge_qs:
            Edge.objects.bulk_update(edge_qs, fields=["type_code"], batch_size=200)

        # 評価表（Markdown）
        md_table = _rated_to_markdown(rated)
        art.markdown_table = md_table
        art.save(update_fields=["markdown_table"])

        _set_status(run, step=2, pct=66, label="処理中", stage_statuses={"Step2": "完了"})

        # ------------------ Step3: 融合モデル（Mermaid + Plotly） ------------------
        _set_status(run, step=3, pct=67, label="処理中", stage_statuses={"Step3": "処理中"})

        # Step3 Mermaid：TYPE 別の矢印スタイル（services.fusion があれば使用）
        rated_for_fusion = [
            {
                "source": r["source"],
                "target": r["target"],
                "effect": r["effect"],
                "prob": r["prob"],
                "sign": r["sign"],
                "type_code": int(r.get("type_code", 1)),
            }
            for r in rated
        ]
        # Step2 の評価結果からノード集合を作成（source/target のユニーク）
        nodes_step2 = sorted({
            _norm_label(r["source"]) for r in rated
        }.union({
            _norm_label(r["target"]) for r in rated
        }))

        # Mermaid（色=sign（+赤/-青）、線種=TYPE、(source,target) 正規化・一意化は fusion 側で実施）
        if fusion_mermaid is not None:
            # 正規化した辞書へ寄せる（fusion 内でも再正規化するが、揺れを最小化）
            rated_for_fusion = [
                {
                    "source": _norm_label(r["source"]),
                    "target": _norm_label(r["target"]),
                    "effect": float(r["effect"]),
                    "prob": float(r["prob"]),
                    "sign": str(r["sign"]),
                    "type_code": int(r.get("type_code", 2)),
                }
                for r in rated
            ]
            mermaid3 = fusion_mermaid(rated_for_fusion)
        else:
            # フォールバック：内蔵ビルダー（※一意化/正規化は内蔵済み）
            mermaid3 = _build_mermaid_step3_colored(
                nodes=nodes_step2,
                rated_edges=[
                    {
                        "source": _norm_label(r["source"]),
                        "target": _norm_label(r["target"]),
                        "effect": float(r["effect"]),
                        "prob": float(r["prob"]),
                        "sign": str(r["sign"]),
                        "type_code": int(r.get("type_code", 2)),
                    }
                    for r in rated
                ],
            )

        art.mermaid_step3 = mermaid3
        art.save(update_fields=["mermaid_step3"])

        # Plotly 出力：階層座標を供給し、TYPE 別 dash を付けたHTMLを保存（utilsが無い場合は簡易にスキップ）
        plot_url = ""
        try:
            # Step2 準拠のノードで配置を作る（VAR 生エッジではなく評価済みエッジで統一）
            # 位置決めに effect/lag を使う高度版がある場合はサービス側に委譲
            pos = _simple_hier_positions(
                nodes=nodes_step2,
                edges=[EdgeEst(
                    source=_norm_label(r["source"]),
                    target=_norm_label(r["target"]),
                    effect=float(r["effect"]),
                    prob=float(r["prob"]),
                    sign=str(r["sign"]),
                    lag=int(r.get("lag", 1)),
                ) for r in rated_for_fusion]
            )
            if export_graph_html:
                plot_url = export_graph_html(
                    rated_edges=rated_for_fusion,
                    positions=pos,
                    out_name=f"run_{run.id}_graph.html",
                )
        except Exception:
            # Plotly 出力が失敗しても Step3 は継続（Mermaid は既に保存済み）
            run.warnings = (run.warnings or []) + ["Step3: Plotly 出力に失敗しました。"]
            run.save(update_fields=["warnings"])

        if plot_url:
            art.plotly_html_path = plot_url
            art.save(update_fields=["plotly_html_path"])

        # すべて成功 → 完了へ
        _set_status(
            run,
            step=3,
            pct=100,
            label="完了",
            stage_statuses={"Step3": "完了"},
            overall="Succeeded",
        )
        run.finished_at = timezone.now()
        run.save(update_fields=["finished_at"])

    except Exception as e:
        # 例外発生 → 失敗へ倒す（ログ/警告も記録）
        tb = traceback.format_exc(limit=6)
        msg = f"Step{run.status.get('step', 1)}失敗: {type(e).__name__}: {e}"
        run.warnings = (run.warnings or []) + [msg, tb]
        run.save(update_fields=["warnings"])
        _set_status(
            run,
            step=run.status.get("step", 1),
            pct=run.status.get("pct", 0),
            label="失敗",
            overall="Failed",
        )
        # ここでraiseしない：UI的には「失敗」で返す
