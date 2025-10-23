# evidra/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),

    # ← スラッシュ無し/有りの両方を用意（フロントは /api/upload-data を呼んでいる）
    path('api/upload-data', views.upload_data),
    path('api/upload-data/', views.upload_data),

    path('api/upload-pdf', views.upload_pdf),
    path('api/upload-pdf/', views.upload_pdf),

    path('api/run', views.create_run),
    path('api/run/', views.create_run),

    path('api/run/<int:run_id>/status', views.run_status),
    path('api/run/<int:run_id>/status/', views.run_status),

    path('api/run/<int:run_id>/artifacts', views.run_artifacts),
    path('api/run/<int:run_id>/artifacts/', views.run_artifacts),

    path('api/chat/<int:run_id>', views.chat),
    path('api/chat/<int:run_id>/', views.chat),

    path('api/chat/<int:run_id>/history', views.chat_history),
    path('api/chat/<int:run_id>/history/', views.chat_history),
    
    path('api/health/aoai', views.health_aoai),
    path('api/health/aoai/', views.health_aoai),
    
    path('api/upload-sample', views.upload_sample)
]
