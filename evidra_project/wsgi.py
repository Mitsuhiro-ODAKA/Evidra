import os
from django.core.wsgi import get_wsgi_application

# WSGI用設定。Gunicorn等での本番運用時のエントリポイント。
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'evidra_project.settings')
application = get_wsgi_application()
