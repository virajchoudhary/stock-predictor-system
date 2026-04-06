from celery import shared_task
from .evolution import evaluate_chromosome
from .models import OptimizedHyperparams, SearchedTicker
import random
import numpy as np

# default watchlist — always optimized regardless of user searches
SEED_WATCHLIST = [
    "AAPL", "TSLA", "SPY", "MSFT", "GOOGL", "GOOG",
    "NIFTYBEES.NS", "RELIANCE.NS", "NVDA", "AMZN", "META", "QQQ"
]


@shared_task(bind=True, max_retries=2)
def run_ga_for_ticker(self, symbol, pop_size=20, generations=15, mutation_rate=0.4):
    """
    Runs the full GA for a single ticker and saves the best result to DB.
    Designed to run as a background Celery task — no streaming, no frontend.
    """
    try:
        population = [
            [
                random.randint(10, 128),
                random.randint(1, 4),
                round(random.uniform(0.0005, 0.05), 4),
                random.randint(10, 60),
                round(random.uniform(0.0, 0.5), 2),
                random.randint(10, 60)
            ]
            for _ in range(pop_size)
        ]

        best_composite   = float('inf')
        best_mse         = float('inf')
        best_accuracy    = 0.0
        best_chromosome  = None

        for gen in range(generations):
            adaptive_mutation = mutation_rate * (1 - gen / generations)
            adaptive_mutation = max(adaptive_mutation, 0.05)  # floor at 5%

            fitnesses, accuracies, composites = [], [], []

            for ind in population:
                try:
                    mse, acc, comp = evaluate_chromosome(ind, symbol)
                except Exception:
                    mse, acc, comp = float('inf'), 0.0, float('inf')
                
                fitnesses.append(mse)
                accuracies.append(acc)
                composites.append(comp)

            sorted_indices = np.argsort(composites)
            population  = [population[i]  for i in sorted_indices]
            fitnesses   = [fitnesses[i]   for i in sorted_indices]
            accuracies  = [accuracies[i]  for i in sorted_indices]
            composites  = [composites[i]  for i in sorted_indices]

            if composites[0] < best_composite:
                best_composite  = composites[0]
                best_mse        = fitnesses[0]
                best_accuracy   = accuracies[0]
                best_chromosome = population[0]

            # next generation
            next_pop = population[:2] if pop_size > 2 else population[:1]

            while len(next_pop) < pop_size:
                idx1, idx2 = random.sample(range(pop_size), 2)
                idx3, idx4 = random.sample(range(pop_size), 2)
                p1 = population[idx1] if composites[idx1] < composites[idx2] else population[idx2]
                p2 = population[idx3] if composites[idx3] < composites[idx4] else population[idx4]

                pt    = random.randint(1, 5)
                child = p1[:pt] + p2[pt:]

                if random.random() < adaptive_mutation:
                    m = random.randint(0, 5)
                    if m == 0: child[0] = random.randint(10, 128)
                    elif m == 1: child[1] = random.randint(1, 4)
                    elif m == 2: child[2] = round(random.uniform(0.0005, 0.05), 4)
                    elif m == 3: child[3] = random.randint(10, 60)
                    elif m == 4: child[4] = round(random.uniform(0.0, 0.5), 2)
                    elif m == 5: child[5] = random.randint(10, 60)

                next_pop.append(child)

            population = next_pop

        # persist
        OptimizedHyperparams.objects.create(
            symbol               = symbol.upper(),
            hidden_size          = best_chromosome[0],
            num_layers           = best_chromosome[1],
            learning_rate        = best_chromosome[2],
            epochs               = best_chromosome[3],
            dropout              = best_chromosome[4],
            seq_len              = best_chromosome[5],
            val_mse              = best_mse,
            directional_accuracy = best_accuracy,
            composite_fitness    = best_composite
        )

    except Exception as exc:
        raise self.retry(exc=exc, countdown=300)


@shared_task
def run_hpo_for_watchlist():
    """
    Triggered nightly by Celery Beat.
    Runs GA for seed tickers + any user-searched tickers.
    """
    from .models import SearchedTicker
    watchlist = list(set(SEED_WATCHLIST + SearchedTicker.get_watchlist()))
    for symbol in watchlist:
        run_ga_for_ticker.delay(symbol)
