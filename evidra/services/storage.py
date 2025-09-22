import os
from pathlib import Path
from django.conf import settings
import re
from uuid import uuid4

# ストレージ抽象化：開発ではローカルファイル、本番はAzure Blob Storage に差し替えを想定。
# ここではローカル保存のみ実装し、戻り値の型をBlob URL互換にしておく。

def save_local_upload(django_file, subdir: str) -> str:
    """
    ユーザーがアップロードしたファイルをローカルディスクに保存し、
    パス（将来はSAS URL）を返す。
    """
    # 保存ディレクトリを構築する（例: uploads/data/ など）
    target_dir = Path(settings.MEDIA_ROOT) / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    # 元のファイル名（パス要素を除去し、危険文字をサニタイズ）
    orig = os.path.basename(django_file.name)
    stem, ext = os.path.splitext(orig)
    # 日本語・英数字・一部記号を許可し、それ以外はアンダースコアに置換
    safe_stem = re.sub(r'[^0-9A-Za-z_\u3040-\u30FF\u4E00-\u9FFF\.\-]+', '_', stem)[:200]
    safe_ext = ext[:10]  # 拡張子は短く制限
    filename = f"{safe_stem}{safe_ext}" if safe_stem else f"upload_{uuid4().hex[:6]}{safe_ext or '.dat'}"
    target_path = target_dir / filename
    # 同名が既にある場合は短いUUID付与でユニーク化
    if target_path.exists():
        filename = f"{safe_stem}_{uuid4().hex[:8]}{safe_ext}"
        target_path = target_dir / filename

    with open(target_path, 'wb') as f:
        for chunk in django_file.chunks():
            f.write(chunk)

    # 返り値は文字列パス。将来はここをBlobのURLに置き換える。
    return str(target_path)

def make_sas_like_local_link(local_path: str) -> str:
    """
    本番ではSAS署名URLを返す。開発中はローカルファイルのfile://リンクは扱いづらいので、
    簡易的に相対パスを返す。実際のダウンロードはNginx等の静的配信で行う想定。
    """
    # 実運用では Azure Blob のSASを発行して返す。
    # 雛形では、/media 以下を配信しないので簡易的にパスを返す（UIでは新規タブで開く）。
    # ※実務ではDjangoでserveせず、オブジェクトストレージの公開URLにすること。
    return local_path
