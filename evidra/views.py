import json
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import pandas as pd

from .models import Dataset, RagDoc, Run, Edge, Artifact, Chat
from .services.storage import save_local_upload, make_sas_like_local_link
from .services.preprocessing import validate_header, guess_frequency
from .tasks import launch_run
from .services.agent import ask_agent

from django.shortcuts import get_object_or_404
from .models import Run, Artifact

MAX_UPLOAD_BYTES = 1_000_000_000  # 1GB

# UIトップ
def index(request):
    # 単純にテンプレートを返す
    return render(request, 'index.html')

# 観測データアップロード
@csrf_exempt
def upload_data(request):
    """観測データ（CSV/XLSX）を受け取り、Dataset レコードを作成する。"""
    if request.method != 'POST':
        return HttpResponseBadRequest("POST only")
    f = request.FILES.get('file')
    if not f:
        return HttpResponseBadRequest("file is required")
    if f.size and f.size > MAX_UPLOAD_BYTES:
        return HttpResponseBadRequest("file too large (max 1GB)")

    # ユーザーの元ファイル名を尊重しつつ安全に保存
    saved_path = save_local_upload(f, subdir="datasets")

    # 先頭行＝列名の想定で読み込む（Excel/CSVを拡張子で分岐）
    p = Path(saved_path)
    try:
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(saved_path)
        elif p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(saved_path, engine='openpyxl')
        else:
            return HttpResponseBadRequest("CSV または XLSX をアップロードしてください。")
    except Exception as e:
        return HttpResponseBadRequest(f"ファイルの読み込みに失敗しました: {e}")

    if df.empty:
        return HttpResponseBadRequest("データが空です。内容をご確認ください。")

    # メタ情報を保存（列名・行数など）
    ds = Dataset.objects.create(
        file_path=str(saved_path),
        columns_json=list(map(str, df.columns)),
        n_rows=int(len(df)),
        freq_guess=""  # 必要なら推定実装
    )
    # プレビュー（先頭10行）。JSON 化のためにプリミティブへ寄せる
    head = df.head(10).copy()
    # 日付等の非プリミティブが混じっても stringify されるように
    head = head.applymap(lambda v: v if (isinstance(v, (int, float, str)) or v is None) else str(v))
    head_records = head.to_dict(orient="records")

    return JsonResponse({
        "dataset_id": ds.id,
        "columns": list(map(str, df.columns)),
        "n_rows": int(len(df)),
    "head_preview": head_records,
        "message": "uploaded"
    })

from .utils.rag import index_pdf_to_cosmos
# RAG PDFアップロード
@csrf_exempt
def upload_pdf(request):
    """RAG用PDFのアップロード（必須ではない）。"""
    if request.method != 'POST':
        return HttpResponseBadRequest("POST only")
    f = request.FILES.get('file')
    if not f:
        return HttpResponseBadRequest("file is required")
    if f.size and f.size > MAX_UPLOAD_BYTES:
        return HttpResponseBadRequest("file too large (max 1GB)")

    saved_path = save_local_upload(f, subdir="rag")
    # ここでは最小限のメタ登録（ページ数などは必要なら別途抽出）
    doc = RagDoc.objects.create(file_path=str(saved_path), pages=0, size_mb=(f.size or 0)/1_000_000)

    return JsonResponse({
        "rag_doc_id": doc.id,
        "pages": doc.pages,
        "size_mb": doc.size_mb,
        "message": "uploaded"
    })

# 実行開始
@csrf_exempt
def create_run(request):
    """
    解析Runを作成して非同期ジョブを起動する。
    - POST JSON: { dataset_id: int, rag_doc_id?: int|null, params?: {...} }
    - 返り値: { run_id: int }
    """
    if request.method != 'POST':
        return HttpResponseBadRequest("POST only")

    try:
        payload = json.loads(request.body.decode('utf-8'))
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")

    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        return HttpResponseBadRequest("dataset_id is required")

    try:
        ds = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        return HttpResponseNotFound("dataset not found")

    rag_doc = None
    rag_doc_id = payload.get("rag_doc_id")
    if rag_doc_id:
        try:
            rag_doc = RagDoc.objects.get(id=rag_doc_id)
        except RagDoc.DoesNotExist:
            rag_doc = None  # RAGは任意なので無視して続行

    params = payload.get("params") or {}
    # 既定値（UIが渡してこない場合の保険）
    params.setdefault("method", "VAR-LiNGAM")
    params.setdefault("lag", 2)
    params.setdefault("boot", 100)
    params.setdefault("seed", 42)
    params.setdefault("preprocessing", {})  # 例: {"fillna":"ffill","diff":false,"standardize":false}

    # Runレコードを作成（初期ステータス）
    run = Run.objects.create(
        dataset=ds,
        rag_doc=rag_doc,
        params=params,
        status={
            "step": 0,
            "pct": 0,
            "label": "未実行",
            "stage_statuses": {"Step1": "未実行", "Step2": "未実行", "Step3": "未実行"},
            "overall": "Pending",
        },
    )
    # 直後に非同期起動
    launch_run(run.id)

    # アーティファクト器を用意（保険）
    Artifact.objects.get_or_create(run=run)

    return JsonResponse({"run_id": run.id})

# ステータス取得
def run_status(request, run_id: int):
    """
    ステータス取得。
    - 返り値 (例): { "run_id": 3, "status": {...}, "warnings": [...] }
    """
    try:
        run = Run.objects.get(id=run_id)
    except Run.DoesNotExist:
        return HttpResponseNotFound("run not found")

    return JsonResponse({
        "run_id": run.id,
        "status": run.status or {},
        "warnings": run.warnings or []
    })

# キャンセル要求
@csrf_exempt
def run_cancel(request, run_id: int):
    try:
        run = Run.objects.get(id=run_id)
    except Run.DoesNotExist:
        return JsonResponse({"detail": "runが見つかりません"}, status=404)

    st = run.status or {}
    st['cancel_requested'] = True
    run.status = st
    run.save(update_fields=['status'])
    return JsonResponse({"ok": True})

# 成果物取得
def run_artifacts(request, run_id: int):
    run = get_object_or_404(Run, id=run_id)
    # OneToOne 前提：常に 1 件だけ存在
    art, _ = Artifact.objects.get_or_create(run=run)

    data = {
        "mermaid_step1": art.mermaid_step1 or "",
        "markdown_table": art.markdown_table or "",
        "mermaid_step3": art.mermaid_step3 or "",
        "plotly_html_path": art.plotly_html_path or "",
    }
    return JsonResponse(data)

# 再実行（完全再現）
@csrf_exempt
def run_replay(request, run_id: int):
    try:
        base = Run.objects.get(id=run_id)
    except Run.DoesNotExist:
        return JsonResponse({"detail": "runが見つかりません"}, status=404)

    # 同一データ/パラメータ/seed=42で新規Runを作成
    new_run = Run.objects.create(
        dataset=base.dataset, rag_doc=base.rag_doc, params=base.params,
        status={"overall":"Running","step":0,"pct":0,"label":"待機中","stage_statuses":{}}
    )
    launch_run(new_run.id)
    return JsonResponse({"run_id": new_run.id, "replay_of": run_id})

@csrf_exempt
def chat(request, run_id: int):
    """Step4: AIエージェントに質問する（Azure OpenAI使用、フォールバック付き）"""
    if request.method != 'POST':
        return HttpResponseBadRequest("POSTのみ許可")
    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return HttpResponseBadRequest("JSONボディが不正です")
    text = (data or {}).get("text", "").strip()
    if not text:
        return HttpResponseBadRequest("textが空です")
    answer = ask_agent(run_id, text)
    return JsonResponse({"answer": answer})


# チャット投稿
@csrf_exempt
def chat_post(request, run_id: int):
    if request.method != 'POST':
        return HttpResponseBadRequest("POSTのみ")
    try:
        run = Run.objects.get(id=run_id)
    except Run.DoesNotExist:
        return JsonResponse({"detail": "runが見つかりません"}, status=404)
    try:
        body = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({"detail": "JSONではありません"}, status=400)
    text = body.get('text') or ""
    if not text.strip():
        return JsonResponse({"ok": False})
    Chat.objects.create(run=run, role='user', text=text)
    return JsonResponse({"ok": True})

# チャットクリア（ソフト削除）
@csrf_exempt
def chat_clear(request, run_id: int):
    try:
        Chat.objects.filter(run_id=run_id).update(soft_deleted=True)
    except Exception:
        return JsonResponse({"ok": False})
    return JsonResponse({"ok": True})
    
from .models import Chat

def _chat_row(c: Chat) -> dict:
    return {"role": c.role, "text": c.text, "created_at": c.created_at.isoformat()}

def chat_history(request, run_id: int):
    """会話履歴を取得（昇順）。GETのみ。"""
    if request.method != 'GET':
        return HttpResponseBadRequest("GETのみ許可")
    qs = Chat.objects.filter(run_id=run_id, soft_deleted=False).order_by('created_at')
    return JsonResponse({"messages": [_chat_row(c) for c in qs]})
    

# ---- Azure OpenAI 接続診断 ----
import os
from typing import Optional
from openai import AzureOpenAI, OpenAI


def health_aoai(request):
    """
    Azure OpenAI の設定と接続を簡易診断してJSONで返す。
    ※キー値そのものは返さず、在/不在や文字数などの最小情報のみを返す。
    """
    def mask(v: Optional[str]) -> str:
        if not v:
            return "MISSING"
        if len(v) <= 6:
            return "***"
        return f"{v[:3]}***{v[-3:]}"  # 先頭3+末尾3だけ

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    # ?dep=xxx を指定したらそれを優先して疎通テスト
    dep = request.GET.get("dep") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    info = {
        "env": {
            "AZURE_OPENAI_ENDPOINT": "SET" if endpoint else "MISSING",
            "AZURE_OPENAI_API_KEY": "SET" if key else "MISSING",
            "AZURE_OPENAI_API_VERSION": api_version,
            "AZURE_OPENAI_CHAT_DEPLOYMENT": dep or "MISSING",
            # 値の一部だけ（漏洩防止）
            "endpoint_sample": mask(endpoint),
            "key_sample": mask(key),
            "deployment_sample": mask(dep),
        },
        "client_init": None,
        "chat_ping": None
    }

    # クライアント初期化（AzureOpenAI → 失敗時は OpenAI(base_url) フォールバック）
    try:
        if not endpoint or not key:
            raise RuntimeError("endpoint_or_key_missing")
        try:
            client = AzureOpenAI(api_key=key, api_version=api_version, azure_endpoint=endpoint)
            info["client_init"] = {"ok": True, "mode": "AzureOpenAI"}
        except Exception as e0:
            if not dep:
                raise
            base_url = endpoint.rstrip("/") + f"/openai/deployments/{dep}"
            client = OpenAI(
                api_key=key,
                base_url=base_url,
                default_query={"api-version": api_version},
                default_headers={"api-key": key},
            )
            info["client_init"] = {"ok": True, "mode": "OpenAI+base_url"}
    except Exception as e:
        info["client_init"] = {"ok": False, "error_type": type(e).__name__, "error": str(e)[:300]}
        return JsonResponse(info)  # ここで返す（以降の呼び出しは無理）

    # チャットの疎通（デプロイ名が必要）
    try:
        if not dep:
            raise RuntimeError("deployment_missing")
        resp = client.chat.completions.create(
            model=dep,
            messages=[
                {"role": "system", "content": "ping"},
                {"role": "user", "content": "pong?"}
            ],
            timeout=10
        )
        txt = (resp.choices[0].message.content or "")[:60]
        info["chat_ping"] = {"ok": True, "sample": txt}
    except Exception as e:
        info["chat_ping"] = {"ok": False, "error_type": type(e).__name__, "error": str(e)[:300]}

    return JsonResponse(info)
