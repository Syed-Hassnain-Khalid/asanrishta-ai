from rest_framework.views import APIView
from rest_framework.response import Response
from django.views import View
from rest_framework import status
from django.contrib.auth import authenticate, get_user_model
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer, LoginSerializer
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
from services import ai_router

User = get_user_model()

class RegisterView(APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "User registered successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data['email']
        password = serializer.validated_data['password']

        user = authenticate(request, email=email, password=password)

        if not user:
            return Response({"error": "Invalid email or password"}, status=status.HTTP_401_UNAUTHORIZED)

        refresh = RefreshToken.for_user(user)
        return Response({
            "refresh": str(refresh),
            "access": str(refresh.access_token),
            "user": {
                "id": user.id,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "gender": user.gender,
                "religion": user.religion,
                "status": user.status,
                "mobile_no": user.mobile_no,
                "marital_status": user.marital_status,
                "city": user.city,
                "occupation": user.occupation,
                "age": user.age,
                "role_id": user.role_id,
                "subscribtionStatus": user.subscribtionStatus,
                "subscribtionExpire": user.subscribtionExpire,
                "connections": user.connections
            }
        }, status=status.HTTP_200_OK)

@method_decorator(csrf_exempt, name='dispatch')
class AskDBView(View):
    def post(self, request, *args, **kwargs):
        try:
            body = json.loads(request.body.decode('utf-8'))
            query = body.get("query")
            user_id = body.get("user_id")

            if not query:
                return JsonResponse({"error": "No query provided"}, status=400)

            # ✅ User detection logic
            if request.user.is_authenticated:
                user = request.user
            elif user_id:
                user = User.objects.get(id=user_id)
            else:
                return JsonResponse({"error": "No authenticated user or user_id provided"}, status=401)

            # ✅ Run query through the agent
            result = ai_router(user, query)
            return JsonResponse({"response": result}, status=200)

        except User.DoesNotExist:
            return JsonResponse({"error": "User not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    def get(self, request, *args, **kwargs):
        return JsonResponse(
            {"message": "Use POST with 'query' and 'user_id' to interact with the agent."},
            status=200
        )