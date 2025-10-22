from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password

User = get_user_model()

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = [
            'email', 'password', 'password2',
            'first_name', 'last_name', 'gender', 'date_of_birth', 'religion',
            'status', 'mobile_no', 'profile_createdby', 'marital_status', 'cast',
            'height', 'education', 'city', 'occupation', 'terms', 'role_id',
            'user_image', 'age', 'userKaTaruf', 'userDiWohtiKaTaruf', 'call_status',
            'subscribtionStatus', 'subscribtionExpire', 'subscribtionDate',
            'connections', 'connectionsDescription', 'tiktokLink'
        ]

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Passwords do not match"})
        return attrs

    def create(self, validated_data):
        validated_data.pop('password2')
        password = validated_data.pop('password')
        user = User.objects.create(**validated_data)
        user.set_password(password)
        user.save()
        return user


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
