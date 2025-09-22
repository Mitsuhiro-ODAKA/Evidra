# evidra/services/agent.py
import os
from typing import Optional, Tuple, List, Dict
from openai import AzureOpenAI, OpenAI
from django.utils import timezone
from ..models import Run, Edge, Chat

# ------------------------------------------------------------
# Azure OpenAI クライアントの生成（失敗理由も返す）
# ------------------------------------------------------------
def _get_client(dep_name: Optional[str]) -> Tuple[Optional[object], Optional[str]]:
    """
    AzureOpenAI クライアントを返す。失敗時は (None, reason)。
    互換モード: AzureOpenAI が初期化できない場合は OpenAI(base_url=.../deployments/{dep}) を使用。
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    # API バージョンは .env で可変にし、未設定なら既定を使う
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not endpoint or not key:
        return None, "endpoint_or_key_missing"
    try:
        cli = AzureOpenAI(api_key=key, api_version=api_version, azure_endpoint=endpoint)
        return cli, None
    except Exception as e:
        err = f"{type(e).__name__}:{str(e)}"
        # フォールバック: OpenAI クライアントに base_url で Azure 形式を指定
        # 注意: dep_name が必要。無い場合はフォールバック不可。
        if dep_name:
            try:
                base_url = endpoint.rstrip("/") + f"/openai/deployments/{dep_name}"
                cli2 = OpenAI(
                    api_key=key,
                    base_url=base_url,
                    default_query={"api-version": api_version},
                    default_headers={"api-key": key},
                )
                return cli2, None
            except Exception as e2:
                return None, f"client_fallback_error:{type(e2).__name__}"
        return None, f"client_init_error:{err}"

# ------------------------------------------------------------
# システムプロンプトと文脈整形
# ------------------------------------------------------------
def _system_prompt() -> str:
    """エージェントの役割を明示するシステムプロンプト。"""
    return (
        "あなたは因果推論の助言エージェントです。"
        "ユーザーのRunに基づく因果グラフ（Step1/3）と評価結果（Step2）を参照し、"
        "根拠（effect, prob, TYPE など）を踏まえて、簡潔かつ正確に回答してください。"
        "不確実な点はその旨を述べ、追加データや検証案を提案してください。"
    )

def _edges_to_bullets(edges: List[Dict]) -> str:
    """エッジ辞書の配列を、人間が読みやすい箇条書き文字列に整形する。"""
    if not edges:
        return "(エッジ情報なし)"
    lines: List[str] = []
    for e in edges:
        src = e.get("source", "-")
        tgt = e.get("target", "-")
        eff = e.get("effect", 0.0)
        prob = e.get("prob", 0.0)
        sgn = e.get("sign", "+")
        tcode = e.get("type_code") or e.get("type")  # 念のため古いキー名にも対応
        if tcode is not None:
            lines.append(f"- {src} -> {tgt} | effect={eff:.3f}, prob={prob:.2f}, sign={sgn}, TYPE={tcode}")
        else:
            lines.append(f"- {src} -> {tgt} | effect={eff:.3f}, prob={prob:.2f}, sign={sgn}")
    return "\n".join(lines)

# ------------------------------------------------------------
# メイン：Step4 のチャット応答
# ------------------------------------------------------------
def ask_agent(run_id: int, user_text: str) -> str:
    """Run に紐づく因果情報を文脈に、Azure OpenAI（あれば）で回答する。"""
    # Run を取得（存在しない場合は即返す）
    try:
        run = Run.objects.get(id=run_id)
    except Run.DoesNotExist:
        return "該当する Run が見つかりませんでした。もう一度実行してください。"

    # 文脈：DB からエッジを読み、箇条書き文字列 context を必ず生成する
    edge_qs = Edge.objects.filter(run=run).values(
        "source", "target", "effect", "prob", "sign", "type_code"
    )
    edges: List[Dict] = list(edge_qs)
    context: str = _edges_to_bullets(edges)

    # まずユーザー発話を保存（会話履歴に残す）
    Chat.objects.create(run=run, role="user", text=user_text, created_at=timezone.now())

    # Azure OpenAI 呼び出し（設定が無い／初期化失敗はフォールバック）
    dep = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    client, reason = _get_client(dep_name=dep)

    if client and dep:
        try:
            messages = [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": f"因果エッジ一覧:\n{context}"},
                {"role": "user", "content": f"質問: {user_text}\n\n"
                                            f"上のエッジ情報を前提に、要点を3〜6行で答えてください。"}
            ]
            resp = client.chat.completions.create(
                model=dep,
                messages=messages,
                timeout=25  # Step4 全体をブロックしない程度のタイムアウト
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # 呼び出し失敗時：理由を付記してフォールバック（鍵・機密は含めない）
            reason2 = f"chat_error:{type(e).__name__}"
            answer = (
                "（フォールバック応答）Azure OpenAI 呼び出しに失敗しました。\n"
                f"reason={reason2}\n\n"
                "以下のエッジを踏まえて検討してください：\n"
                f"{context}\n\n"
                "・TYPE2 は整合的と評価されています。\n"
                "・prob（ブートストラップ頻度）と effect の大きさで優先度を判断してください。"
            )
    else:
        # クライアント未初期化／デプロイ名未設定など：理由を明示してフォールバック
        missing = []
        if not client: missing.append(f"client={reason}")
        if not dep:    missing.append("deployment=missing")
        m = "; ".join(missing) if missing else "unknown"
        answer = (
            "（フォールバック応答）Azure OpenAI が使用できません。\n"
            f"reason={m}\n\n"
            "以下のエッジを踏まえて検討してください：\n"
            f"{context}\n\n"
            "・.env の AZURE_OPENAI_ENDPOINT / API_KEY / CHAT_DEPLOYMENT / API_VERSION をご確認ください。"
        )

    # アシスタント応答を保存（会話履歴に残す）
    Chat.objects.create(run=run, role="assistant", text=answer, created_at=timezone.now())
    return answer
