from django.contrib import admin
from .models import PredictionRecord

@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ('disease_type', 'result', 'confidence', 'created_at')
    list_filter = ('disease_type', 'created_at')
    search_fields = ('result',)