#!/usr/bin/env python
import os
import sys
from dotenv import load_dotenv; load_dotenv()

def main():
    # Djangoの設定モジュールを指す環境変数を設定する
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'evidra_project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        # 依存関係の未インストール時に分かりやすいエラーを出す
        raise ImportError(
            "Djangoが見つかりません。仮想環境とrequirementsを確認してください。"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
