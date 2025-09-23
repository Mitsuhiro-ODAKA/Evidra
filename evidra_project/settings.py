import os
from pathlib import Path
import dj_database_url
from dotenv import load_dotenv; load_dotenv()

# BASE_DIR はプロジェクトのルートパスを指す
BASE_DIR = Path(__file__).resolve().parent.parent

# 環境変数の読み込み（.envは任意）
from dotenv import load_dotenv
load_dotenv()

# 開発向けの秘密鍵（本番は環境変数で管理する）
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'evidra-secret-key')

# 開発中はDEBUG=True。本番はFalseにすること
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
# DEBUG = True

# 許可するホスト名（開発中はワイルドカードでも良い）
# ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
ALLOWED_HOSTS = ["*"]
RENDER_EXTERNAL_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if RENDER_EXTERNAL_HOSTNAME:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)

# CSRF 対応（Render の https://<name>.onrender.com）
CSRF_TRUSTED_ORIGINS = [
    f"https://{RENDER_EXTERNAL_HOSTNAME}" for _ in [0] if RENDER_EXTERNAL_HOSTNAME
]

# Djangoアプリの登録
INSTALLED_APPS = [
    'django.contrib.admin',        # 管理サイト（本雛形では未使用だが有用）
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'evidra',                      # 本アプリ
]

# ミドルウェア設定
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    "whitenoise.middleware.WhiteNoiseMiddleware",
    'django.contrib.sessions.middleware.SessionMiddleware',  # セッション管理（チャットUI等で使用）
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',            # CSRF保護（APIは基本POST想定）
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# ルートURL設定
ROOT_URLCONF = 'evidra_project.urls'

# テンプレート設定（`templates/`を検索対象にする）
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],  # アプリ内templatesを使うため空でOK
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',  # requestをテンプレに渡す
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI設定
WSGI_APPLICATION = 'evidra_project.wsgi.application'

# データベース設定（開発用SQLite。本番はCosmosをservices.storageで扱う想定）
# DB：DATABASE_URL があれば使う（Render の Postgres）
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DATABASES = {"default": dj_database_url.parse(DATABASE_URL, conn_max_age=600)}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": str(BASE_DIR / "db.sqlite3"),
        }
    }

# パスワードバリデータ（開発雛形では標準設定）
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]

# ロケール/タイムゾーン（ユーザーは日本在住のためAsia/Tokyo）
LANGUAGE_CODE = 'ja'
TIME_ZONE = 'Asia/Tokyo'
USE_I18N = True
USE_TZ = True

# --- ここから静的/メディア設定を置き換え ---

# 静的ファイルのURL
STATIC_URL = "/static/"

# 静的ファイルの探索ディレクトリ（アプリ内とプロジェクト直下の両方を見る）
STATICFILES_DIRS = [
    BASE_DIR / "evidra" / "static",   # <app>/static
    BASE_DIR / "static",              # プロジェクト直下（存在すれば）
]

# 収集先（本番/Renderで collectstatic するときに使用）
STATIC_ROOT = BASE_DIR / "staticfiles"

# メディア（ユーザーアップロード）
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# Whitenoise の開発/本番切り分け
if DEBUG:
    # 開発中は Finder を使って最新ファイルをそのまま配信（ハッシュ不要）
    WHITENOISE_AUTOREFRESH = True
    WHITENOISE_USE_FINDERS = True
    # ストレージはデフォルトのまま（Manifest にしない）
    STATICFILES_STORAGE = "whitenoise.storage.CompressedStaticFilesStorage"
else:
    # 本番は Manifest（ハッシュ）でキャッシュ効かせる
    STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"


# セッション（チャットや実行状態の軽い保持に利用）
SESSION_ENGINE = 'django.contrib.sessions.backends.db'

# CORSやCSRFの微調整は必要に応じて追加
