from django.db import models
from django.utils import timezone
from datetime import timedelta


class OptimizedHyperparams(models.Model):
    symbol               = models.CharField(max_length=10)
    hidden_size          = models.IntegerField()
    num_layers           = models.IntegerField()
    learning_rate        = models.FloatField()
    epochs               = models.IntegerField()
    val_mse              = models.FloatField()
    directional_accuracy = models.FloatField()
    composite_fitness    = models.FloatField()
    created_at           = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    @classmethod
    def get_valid_cache(cls, symbol):
        cutoff = timezone.now() - timedelta(days=3)
        return cls.objects.filter(symbol=symbol.upper(), created_at__gte=cutoff).first()
