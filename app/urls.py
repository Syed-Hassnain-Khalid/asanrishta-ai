from django.urls import path
from .views import RegisterView, LoginView, AskDBView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path("ask-db/", AskDBView.as_view(), name="ask_db"),
]
