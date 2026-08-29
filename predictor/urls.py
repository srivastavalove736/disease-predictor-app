from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    
    # Prediction Endpoints
    path('predict/diabetes/', views.predict_diabetes, name='predict_diabetes'),
    path('predict/heart/', views.predict_heart, name='predict_heart'),
    path('predict/skin/', views.predict_skin, name='predict_skin'),
    path('predict/parkinsons/', views.predict_parkinsons, name='predict_parkinsons'),
    path('predict/brain/', views.predict_brain, name='predict_brain'),
    path('predict/report/', views.predict_report, name='predict_report'),
    
    # History & API Endpoints
    path('history/', views.history, name='history'),
    path('download-pdf/<int:record_id>/', views.download_pdf, name='download_pdf'),
    path('api/ai-chat/', views.ai_chat_api, name='ai_chat_api'),
]