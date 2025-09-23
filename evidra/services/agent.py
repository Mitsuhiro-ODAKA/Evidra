# evidra/services/agent.py
from __future__ import annotations
import os
from typing import Optional, Tuple, List, Dict
from django.utils import timezone
from openai import AzureOpenAI

from ..models import Run, Edge, Chat

# ---------------------------
# Azure OpenAI クライアント
# ---------------------------

def _get_client() -> Tuple[Optional[AzureOpenAI], Optional[str]]:
    """
    Azure OpenAI クライアントを返す。失敗時は (None, reason)。
    必須ENV: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    任意ENV: AZURE_OPENAI_API_VERSION（既定: 2024-06-01）
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
    if not endpoint or not key:
        return None, "endpoint_or_key_missing"
    try:
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)
        return client, None
    except Exception as e:
        return None, f"client_init_error:{type(e).__name__}"

# ---------------------------
# プロンプト/コンテキスト
# ---------------------------

def _system_prompt() -> str:
    """エージェントの役割を明示するシステムプロンプト。"""
    return (
        "あなたは因果グラフに基づいて短く実務的に助言するアシスタントです。"
        "次の前提で答えてください：\n"
        "1) TYPE2は最も整合的。TYPE1は棄却。TYPE3/4/5は注意深い解釈が必要。\n"
        "2) prob（ブート出現頻度）と |effect| でエッジの優先度を概観する。\n"
        "3) 出力は3〜6行の短文で。"
    )

def _edges_to_bullets(edges: List[Dict]) -> str:
    """エッジ配列を読みやすい箇条書きへ整形。"""
    if not edges:
        return "(エッジが見つかりません)"
    lines: List[str] = []
    for e in edges:
        src = e.get("source", "-")
        tgt = e.get("target", "-")
        eff = float(e.get("effect", 0.0))
        prob = float(e.get("prob", 0.0))
        sgn = str(e.get("sign", "+"))
        tcode = e.get("type_code")
        ttxt = f"TYPE{tcode}" if tcode is not None else "TYPE?"
        lines.append(f"- {src} -> {tgt}  [{ttxt}, sign={sgn}, effect={eff:.3f}, prob={prob:.2f}]")
    return "\n".join(lines)

# ---------------------------
# Public: エージェント回答
# ---------------------------

def ask_agent(run_id: int, user_text: str) -> str:
    """
    Step4: DB上の因果エッジを前提に Azure OpenAI へ質問。
    - 失敗時は理由を明示したフォールバックを返す。
    - 成否に関わらず会話履歴(Chat)に保存。
    """
    try:
        run = Run.objects.get(id=run_id)
    except Run.DoesNotExist:
        return "Runが見つかりません。まず解析を実行してください。"

    # ユーザー発話を保存
    Chat.objects.create(run=run, role="user", text=user_text, created_at=timezone.now())

    # 文脈（最新のエッジを取得）
    edges = list(Edge.objects.filter(run=run).values("source", "target", "effect", "prob", "sign", "type_code"))
    context = _edges_to_bullets(edges)

    # Azure クライアント初期化
    client, reason = _get_client()
    dep = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")  # Azureの「デプロイ名」

    if not client or not dep:
        missing = []
        if not client: missing.append(f"client={reason}")
        if not dep:    missing.append("deployment=missing")
        msg = (
            "（フォールバック応答）Azure OpenAI が使用できません。\n"
            f"reason={'; '.join(missing) or 'unknown'}\n\n"
            f"以下のエッジを前提に、質問「{user_text}」を検討してください：\n{context}"
        )
        Chat.objects.create(run=run, role="assistant", text=msg, created_at=timezone.now())
        return msg

    # 正常系：Chat Completions
    try:
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": f"因果エッジ一覧:\n{context}"},
            {"role": "user", "content": f"質問: {user_text}\n\n上のエッジ情報を前提に、要点を3〜6行で答えてください。"},
        ]
        resp = client.chat.completions.create(
            model=dep,
            messages=messages,
            temperature=0.2,
            timeout=25,
        )
        answer = (resp.choices[0].message.content or "").strip()
        Chat.objects.create(run=run, role="assistant", text=answer, created_at=timezone.now())
        return answer

    except Exception as e:
        reason2 = f"chat_error:{type(e).__name__}"
        msg = (
            "（フォールバック応答）Azure OpenAI 呼び出しに失敗しました。\n"
            f"reason={reason2}\n\n"
            f"- 質問: {user_text}\n"
            "- TYPE2を優先し、prob と |effect| で重要度を判断してください。\n\n"
            f"【参照エッジ】\n{context}"
        )
        Chat.objects.create(run=run, role="assistant", text=msg, created_at=timezone.now())
        return msg
