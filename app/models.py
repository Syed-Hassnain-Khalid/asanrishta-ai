from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.conf import settings


class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('Email is required')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        return self.create_user(email, password, **extra_fields)


class CustomUser(AbstractUser):
    username = None  # disable username
    email = models.EmailField(unique=True)

    gender = models.CharField(max_length=50, null=True, blank=True)
    date_of_birth = models.DateTimeField(null=True, blank=True)
    religion = models.CharField(max_length=50, null=True, blank=True)
    status = models.CharField(max_length=50, null=True, blank=True)
    mobile_no = models.CharField(max_length=50, null=True, blank=True)
    profile_createdby = models.IntegerField(null=True, blank=True)
    marital_status = models.CharField(max_length=50, null=True, blank=True)
    cast = models.CharField(max_length=50, null=True, blank=True)
    height = models.CharField(max_length=50, null=True, blank=True)
    education = models.CharField(max_length=50, null=True, blank=True)
    city = models.IntegerField(null=True, blank=True)
    occupation = models.CharField(max_length=50, null=True, blank=True)
    terms = models.BooleanField(null=True, blank=True)
    role_id = models.IntegerField(null=True, blank=True)
    user_image = models.TextField(null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    userKaTaruf = models.TextField(null=True, blank=True)
    userDiWohtiKaTaruf = models.TextField(null=True, blank=True)
    call_status = models.TextField(null=True, blank=True)
    created_date = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_date = models.DateTimeField(auto_now=True, null=True, blank=True)
    subscribtionStatus = models.CharField(max_length=50, null=True, blank=True)
    subscribtionExpire = models.DateTimeField(null=True, blank=True)
    subscribtionDate = models.CharField(max_length=50, null=True, blank=True)
    connections = models.IntegerField(null=True, blank=True)
    connectionsDescription = models.TextField(null=True, blank=True)
    tiktokLink = models.TextField(null=True, blank=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []  # email and password are required by default

    objects = CustomUserManager()

    class Meta:
        db_table = 'Users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def __str__(self):
        return self.email

class ConversationMemory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='memories')
    user_input = models.TextField()
    ai_response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.user.email} - {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
    