# evidra/services/agent.py
from __future__ import annotations
import os
from typing import Optional, Tuple, List, Dict
from django.utils import timezone
from openai import AzureOpenAI
from ..models import Run, Edge, Chat

SYSTEM_PROMPT = (
    "あなたは因果グラフに基づいて短く実務的に助言するアシスタントです。"
    "次の前提で答えてください：\n"
    "1) TYPE2は最も整合的。TYPE1は棄却。TYPE3/4/5は注意深い解釈が必要。\n"
    "2) prob（ブート出現頻度）と |effect| でエッジの優先度を概観する。\n"
    "3) 出力は3〜6行の短文で。"
)

def _get_client() -> Tuple[Optional[AzureOpenAI], Optional[str]]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    if not endpoint or not key:
        return None, "endpoint_or_key_missing"
    try:
        cli = AzureOpenAI(api_key=key, api_version=api_version, azure_endpoint=endpoint)
        return cli, None
    except Exception as e:
        return None, f"client_init_error:{type(e).__name__}"

def _edges_to_bullets(run_id: int, limit_rows: int = 60, limit_chars: int = 6000) -> str:
    """DBのエッジを箇条書きへ。過長入力を防ぐため件数/文字数を制限。"""
    rows = (
        Edge.objects
        .filter(run_id=run_id)
        .order_by("-prob", "-effect")
        .values("source", "target", "effect", "prob", "sign", "type_code")[:limit_rows]
    )
    lines: List[str] = []
    for e in rows:
        tp = int(e["type_code"]) if e["type_code"] is not None else 2
        lines.append(
            f"- {e['source']} -> {e['target']} "
            f"(TYPE{tp}, sign={e.get('sign','?')}, effect={float(e.get('effect',0)):.3f}, prob={float(e.get('prob',0)):.2f})"
        )
        if sum(len(x) + 1 for x in lines) > limit_chars:
            break
    return "\n".join(lines) if lines else "(エッジが保存されていません)"

def ask_agent(run_id: int, user_text: str) -> str:
    # ユーザー発言は必ず保存
    try:
        run = Run.objects.get(id=run_id)
    except Run.DoesNotExist:
        return "Runが見つかりません。まず解析を実行してください。"
    Chat.objects.create(run=run, role="user", text=user_text, created_at=timezone.now())

    # 文脈作成（制限付き）
    context = _edges_to_bullets(run_id)

    # Azure OpenAI
    client, reason = _get_client()
    dep = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")  # ★Azureの“デプロイ名”を指定
    if not client or not dep:
        msg = (
            "（フォールバック応答）Azure OpenAI が使用できません。\n"
            f"reason={('client='+str(reason) if not client else '')}"
            f"{'; ' if (not client and not dep) else ''}"
            f"{'deployment=missing' if not dep else ''}\n\n"
            f"利用可能な因果エッジ:\n{context}\n\n質問: {user_text}"
        )
        Chat.objects.create(run=run, role="assistant", text=msg, created_at=timezone.now())
        return msg

    try:
        resp = client.chat.completions.create(
            model=dep,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"利用可能な因果エッジ:\n{context}"},
                {"role": "user", "content": f"質問: {user_text}\n上のエッジ情報（とTYPE解釈）を踏まえて、3〜6行で答えてください。"},
            ],
            timeout=25,
        )
        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("empty_chat_response")
        Chat.objects.create(run=run, role="assistant", text=answer, created_at=timezone.now())
        return answer

    except Exception as e:
        # BadRequest(400)含む全例外をフォールバック表示に集約
        et = type(e).__name__
        msg = (
            "（フォールバック応答）Azure OpenAI 呼び出しに失敗しました。\n"
            f"reason=chat_error:{et}\n\n"
            f"利用可能な因果エッジ:\n{context}\n\n質問: {user_text}"
        )
        Chat.objects.create(run=run, role="assistant", text=msg, created_at=timezone.now())
        return msg
