# evidra/tasks.py
from __future__ import annotations

import math
import json
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import threading
import re
import time

import numpy as np
import pandas as pd
from django.utils import timezone

from .models import Run, Dataset, RagDoc, Edge, Artifact

try:
    from .services.validation import build_markdown_from_rated as build_md_llm
except Exception:
    build_md_llm = None
    
def _safe_rate_edges(run, edges_dicts, rag_doc):
    """
    services.validation.rate_edges を、安全に呼び出す薄いラッパ。
    失敗時は None を返す（呼び元でフォールバックへ）。
    """
    try:
        if rate_edges_llm is None:
            return None
        # rag_doc が None でも関数にそのまま渡してOK（内部で rag_present 判定）
        return rate_edges_llm(run, edges_dicts, rag_doc=rag_doc)
    except Exception as _e:
        # ここで落とさず None を返す
        return None

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
    
def _call_with_timeout(fn, timeout_sec: int, *args, **kwargs):
    """関数を別スレッドで実行し、timeout を超えたら None を返す簡易ラッパ。"""
    out = {"done": False, "result": None, "err": None}
    def _runner():
        try:
            out["result"] = fn(*args, **kwargs)
            out["done"] = True
        except Exception as e:
            out["err"] = e
    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)
    if not out["done"] and out["err"] is None:
        return None  # タイムアウト
    if out["err"] is not None:
        raise out["err"]
    return out["result"]

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
            # "citations": "-",           # RAG 実装時は "DOC#pX, DOC#pY" のように埋める
        })
    return rated


def _rated_to_markdown(rated: List[dict]) -> str:
    """評価結果を Markdown テーブルへ整形（UI 側で HTML <table> へレンダリング）。"""
    headers = [
        "source", "target", "effect", "prob",
        "因果の有無", "因果の向き", "因果の正負", "正負(推定)", "TYPE"
        # , "citations"
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
            f"TYPE{int(r.get('type_code', 1))}"
            # , str(r.get("citations", "-")),
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

    # ------------------ Step1: 因果探索 ------------------
    _set_status(run, step=1, pct=10, label="処理中", stage_statuses={"Step1": "処理中"})
    try:
        df = _load_observation_frame(run.dataset)

        # 日付列の検出とソート
        tcol = _choose_time_column(df)
        if tcol:
            dt = pd.to_datetime(df[tcol], errors="coerce")
            if dt.notna().sum() < max(10, int(0.5 * len(df))):
                raise ValueError(f"日付列 '{tcol}' の解釈に失敗（有効行が少なすぎます）")
            df = df.loc[dt.notna()].copy()
            df.sort_values(by=tcol, inplace=True)
            df.reset_index(drop=True, inplace=True)

        # 対象列抽出・数値化
        var_cols = [c for c in df.columns if c != tcol]
        if not var_cols:
            raise ValueError("数値解析対象の列がありません（日時列しか残っていません）")
        df_num = _coerce_numeric(df[var_cols], ignore_cols=[])
        if df_num.isna().all(axis=None):
            raise ValueError("数値化後にすべて欠損になりました（列名やデータ型をご確認ください）")

        # パラメータ
        params = run.params or {}
        lag = int(params.get("lag", 2))
        boot = int(params.get("boot", 100))
        seed = int(params.get("seed", 42))
        edge_thr = float(params.get("edge_threshold", 1e-8))
        if lag >= len(df_num) - 5:
            raise ValueError(f"ラグ p={lag} が系列長に対して大きすぎます（必要: p <= T-5, T={len(df_num)}）")

        if VARLiNGAM is None:
            raise RuntimeError("lingam が見つかりません。`pip install lingam statsmodels scikit-learn scipy` を実行してください。")

        # 推定
        edges, _all_nodes = _var_lingam_edges(
            df=df_num, var_cols=var_cols, lag=lag, boot=boot, seed=seed,
            standardize=True, thr=edge_thr,
        )
        if not edges:
            raise ValueError("有意なエッジが 0 件でした。edge_threshold を緩める、ラグ/ブート回数を調整してください。")

        # Edge保存（既存クリア）
        Edge.objects.filter(run=run).delete()
        Edge.objects.bulk_create([
            Edge(run=run,
                 source=_norm_label(e.source),
                 target=_norm_label(e.target),
                 effect=e.effect,
                 prob=e.prob,
                 sign=e.sign,
                 type_code=None)
            for e in edges
        ])

        # 要件により Step1 の Mermaid は非表示に（空文字を保存）
        art.mermaid_step1 = ""
        art.save(update_fields=["mermaid_step1"])

        _set_status(run, step=1, pct=33, label="処理中", stage_statuses={"Step1": "完了"})

    except Exception as e:
        # Step1 失敗 → 失敗で終了
        run.warnings = (run.warnings or []) + [f"Step1で失敗: {type(e).__name__}: {e}"]
        run.save(update_fields=["warnings"])
        _set_status(run, step=1, pct=0, label="失敗", overall="Failed")
        run.finished_at = timezone.now()
        run.save(update_fields=["finished_at"])
        return  # <<<<<< ここで終了

    # ------------------ Step2: 背景知識による妥当性確認 ------------------
    _set_status(run, step=2, pct=34, label="処理中", stage_statuses={"Step2": "処理中"})
    try:
        # DBに保存済みの Step1 エッジを dict 化（評価器が dict を想定）
        edges_qs = list(Edge.objects.filter(run=run))
        edges_dicts = [{
            "source": _norm_label(x.source),
            "target": _norm_label(x.target),
            "effect": float(x.effect),
            "prob": float(x.prob),
            "sign": ("+" if float(x.effect) >= 0 else "-"),
            "lag": 1,
        } for x in edges_qs]

        rated = _safe_rate_edges(run, edges_dicts, run.rag_doc)
        if rated is None:
            # LLM 未設定/失敗など → フォールバック
            rated = _rate_edges_simple([
                EdgeEst(e["source"], e["target"], e["effect"], e["prob"],
                        "+" if e["effect"] >= 0 else "-", e["lag"])
                for e in edges_dicts
            ])
            run.warnings = (run.warnings or []) + ["Step2: LLM/RAG unavailable → fallback"]
            run.save(update_fields=["warnings"])

        # TYPE を Edge レコードに反映（キー衝突なしで安全に更新）
        typemap = {(r["source"], r["target"]): int(r.get("type_code", 2)) for r in rated}
        to_update = []
        for eo in edges_qs:
            key = (_norm_label(eo.source), _norm_label(eo.target))
            if key in typemap:
                eo.type_code = typemap[key]
                to_update.append(eo)
        if to_update:
            Edge.objects.bulk_update(to_update, fields=["type_code"], batch_size=200)

        # 評価表（Markdown）— rated が空でも空表にはならずヘッダだけ出る想定
        if rate_edges_llm is not None and build_md_llm is not None:
            try:
                md_table = build_md_llm(rated)   # ← eval_has/dir/sign と citations を正しく出力
            except Exception:
                md_table = _rated_to_markdown(rated)  # フォールバック
        else:
            md_table = _rated_to_markdown(rated)
        art.markdown_table = md_table
        art.save(update_fields=["markdown_table"])

        _set_status(run, step=2, pct=66, label="処理中", stage_statuses={"Step2": "完了"})

    except Exception as e:
        # ここで落ちても致命にしない：フォールバックで表だけ出す → 完了扱いへ
        run.warnings = (run.warnings or []) + [f"Step2 soft-fail: {type(e).__name__}: {e}"]
        run.save(update_fields=["warnings"])
        try:
            # 表だけでも出しておく
            md_table = _rated_to_markdown(_rate_edges_simple([
                EdgeEst(_norm_label(x.source), _norm_label(x.target),
                        float(x.effect), float(x.prob),
                        "+" if float(x.effect) >= 0 else "-", 1)
                for x in Edge.objects.filter(run=run)
            ]))
            art.markdown_table = md_table
            art.save(update_fields=["markdown_table"])
        except Exception:
            pass
        # 失敗には倒さず「完了」に進める（Step3 はそのまま続行できる）
        _set_status(run, step=2, pct=66, label="処理中", stage_statuses={"Step2": "完了"})

    # ------------------ Step3: 融合モデル（Mermaid / Plotly） ------------------
    _set_status(run, step=3, pct=67, label="処理中", stage_statuses={"Step3": "処理中"})
    try:
        # Step2 の評価結果ベースでノード/エッジを作る
        rated_for_fusion = [{
            "source": e.source, "target": e.target,
            "effect": e.effect, "prob": e.prob,
            "sign": e.sign, "type_code": (e.type_code or 2),
        } for e in Edge.objects.filter(run=run)]

        # Mermaid（services.fusion があれば使用）
        if fusion_mermaid:
            mermaid3 = fusion_mermaid(rated_for_fusion)
        else:
            nodes_step2 = sorted({r["source"] for r in rated_for_fusion} | {r["target"] for r in rated_for_fusion})
            mermaid3 = _build_mermaid_step3_colored(nodes_step2, rated_for_fusion)

        art.mermaid_step3 = mermaid3
        art.save(update_fields=["mermaid_step3"])

        # Plotly（任意）
        try:
            if export_graph_html:
                nodes_step2 = sorted({r["source"] for r in rated_for_fusion} | {r["target"] for r in rated_for_fusion})
                pos = _simple_hier_positions(
                    nodes=nodes_step2,
                    edges=[EdgeEst(r["source"], r["target"], r["effect"], r["prob"], r["sign"], 1) for r in rated_for_fusion]
                )
                url = export_graph_html(rated_edges=rated_for_fusion, positions=pos, out_name=f"run_{run.id}_graph.html")
                if url:
                    art.plotly_html_path = url
                    art.save(update_fields=["plotly_html_path"])
        except Exception:
            run.warnings = (run.warnings or []) + ["Step3: Plotly 出力に失敗しました。"]
            run.save(update_fields=["warnings"])

        _set_status(run, step=3, pct=100, label="完了", stage_statuses={"Step3": "完了"}, overall="Succeeded")
        run.finished_at = timezone.now()
        run.save(update_fields=["finished_at"])

    except Exception as e:
        tb = traceback.format_exc(limit=6)
        run.warnings = (run.warnings or []) + [f"Step3で失敗: {type(e).__name__}: {e}", tb]
        run.save(update_fields=["warnings"])
        _set_status(run, step=3, pct=0, label="失敗", overall="Failed")
        run.finished_at = timezone.now()
        run.save(update_fields=["finished_at"])
