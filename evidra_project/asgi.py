import os
from django.core.asgi import get_asgi_application

# ASGI用設定。将来的にWebSocket/SSEに拡張する場合の入口となる。
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'evidra_project.settings')
application = get_asgi_application()
