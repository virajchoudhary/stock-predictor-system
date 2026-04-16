from django.core.management.base import BaseCommand
from api.models import OptimizedHyperparams
from django.utils import timezone
from datetime import timedelta

class Command(BaseCommand):
    help = 'Clear OptimizedHyperparams records older than N days'

    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, default=3)
        parser.add_argument('--symbol', type=str, default=None)

    def handle(self, *args, **options):
        if options['symbol']:
            deleted, _ = OptimizedHyperparams.objects.filter(
                symbol=options['symbol'].upper()
            ).delete()
            self.stdout.write(f"Deleted {deleted} records for {options['symbol']}")
        else:
            cutoff = timezone.now() - timedelta(days=options['days'])
            deleted, _ = OptimizedHyperparams.objects.filter(
                created_at__lt=cutoff
            ).delete()
            self.stdout.write(f"Deleted {deleted} old records")
