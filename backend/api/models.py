from django.db import models
from django.utils import timezone
from datetime import timedelta


class OptimizedHyperparams(models.Model):
    symbol               = models.CharField(max_length=10)
    hidden_size          = models.IntegerField()
    num_layers           = models.IntegerField()
    learning_rate        = models.FloatField()
    epochs               = models.IntegerField()
    dropout              = models.FloatField(default=0.0)
    seq_len              = models.IntegerField(default=20)
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

    @classmethod
    def clear_cache(cls, symbol):
        cls.objects.filter(symbol=symbol.upper()).delete()


class SearchedTicker(models.Model):
    symbol      = models.CharField(max_length=10, unique=True)
    search_count = models.IntegerField(default=1)
    last_searched = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-search_count']

    @classmethod
    def log(cls, symbol):
        """
        Upsert ticker into watchlist. Increment count if exists.
        If total tickers exceed 20, drop the least searched.
        """
        obj, created = cls.objects.get_or_create(symbol=symbol.upper())
        if not created:
            obj.search_count += 1
            obj.save(update_fields=['search_count', 'last_searched'])
        # Cap at 20 most searched
        total = cls.objects.count()
        if total > 20:
            least = cls.objects.order_by('search_count').first()
            if least and least.symbol != symbol.upper():
                least.delete()

    @classmethod
    def get_watchlist(cls):
        return list(cls.objects.values_list('symbol', flat=True))
