import json
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from .ai_services import LSTMModel, get_price_history


def get_train_val_data(symbol="AAPL", seq_len=20):
    import ta
    df = get_price_history(symbol, period="2y")
    if df is None or df.empty:
        raise ValueError(f"Not enough data for {symbol}")

    close = df['Close']
    high  = df['High']
    low   = df['Low']

    # 5 technical features — same ones used in TrendPredictor
    df['rsi']        = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df['macd_diff']  = ta.trend.MACD(close=close).macd_diff()
    df['bb_width']   = (
        ta.volatility.BollingerBands(close=close, window=20).bollinger_hband() -
        ta.volatility.BollingerBands(close=close, window=20).bollinger_lband()
    )
    df['atr']        = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    sma20            = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df['sma_dist']   = (close - sma20) / sma20

    feature_cols = ['Close', 'rsi', 'macd_diff', 'bb_width', 'atr', 'sma_dist']
    df = df[feature_cols].dropna()

    scaler      = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df.values)
    input_size  = data_scaled.shape[1]  # 6

    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i + seq_len])
        y.append(data_scaled[i + seq_len, 0])  # predict Close only

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    train_size       = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val     = X[train_size:], y[train_size:]

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val,   dtype=torch.float32),
        torch.tensor(y_val,   dtype=torch.float32),
        scaler,
        input_size
    )


def evaluate_chromosome(chromosome, data_tensors, input_size):
    hidden_size, num_layers, lr, epochs = chromosome
    X_train, y_train, X_val, y_val = data_tensors

    # Initialize the model using the same class used in our standard predictions
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Validation and Accuracy Calculation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

        # Directional Accuracy — did the model predict the correct price direction?
        # X_val shape is (batch, seq_len, input_size); column 0 is Close
        prev_prices = X_val[:, -1, 0]  # the last known Close before prediction

        actual_deltas    = y_val.squeeze() - prev_prices
        predicted_deltas = val_outputs.squeeze() - prev_prices

        actual_signs    = torch.sign(actual_deltas)
        predicted_signs = torch.sign(predicted_deltas)

        correct  = (actual_signs == predicted_signs).sum().item()
        total    = len(actual_signs)
        accuracy = (correct / total) * 100 if total > 0 else 0.0

    # Composite fitness: penalises high MSE AND low accuracy. Lower is better.
    composite = val_loss.item() * (1 - (accuracy / 100))
    return val_loss.item(), accuracy, composite


def evolutionary_hpo_generator(symbol="AAPL", pop_size=5, generations=5, mutation_rate=0.2):
    """
    Generator function that evaluates an entire population generation by generation.
    It yields the state of the EA back to the caller as JSON newline streams.
    """
    try:
        result       = get_train_val_data(symbol, seq_len=20)
        data_tensors = result[:4]
        input_size   = result[5]
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n\n"
        return

    # Chromosome representation:
    # Index 0: hidden_size (10 to 100)
    # Index 1: num_layers (1 to 3)
    # Index 2: lr (0.001 to 0.05)
    # Index 3: epochs (10 to 50)
    population = []
    for _ in range(pop_size):
        population.append([
            random.randint(10, 100),
            random.randint(1, 3),
            round(random.uniform(0.001, 0.05), 4),
            random.randint(10, 50)
        ])

    best_overall_composite   = float('inf')
    best_overall_fitness_mse = None
    best_overall_accuracy    = None
    best_overall_chromosome  = None

    for gen in range(1, generations + 1):
        fitnesses  = []
        accuracies = []
        composites = []
        for ind in population:
            loss, acc, composite = evaluate_chromosome(ind, data_tensors, input_size)
            fitnesses.append(loss)
            accuracies.append(acc)
            composites.append(composite)

        # Sort population by composite fitness (lower is better)
        sorted_indices = np.argsort(composites)
        population = [population[i] for i in sorted_indices]
        fitnesses  = [fitnesses[i]  for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        composites = [composites[i] for i in sorted_indices]

        current_best_loss      = fitnesses[0]
        current_best_acc       = accuracies[0]
        current_best_composite = composites[0]
        current_best_ind       = population[0]

        if current_best_composite < best_overall_composite:
            best_overall_composite   = current_best_composite
            best_overall_fitness_mse = current_best_loss
            best_overall_accuracy    = current_best_acc
            best_overall_chromosome  = current_best_ind

        yield json.dumps({
            "generation": gen,
            "best_loss": current_best_loss,
            "best_accuracy": current_best_acc,
            "best_composite": current_best_composite,
            "best_chromosome": {
                "hidden_size": current_best_ind[0],
                "num_layers": current_best_ind[1],
                "learning_rate": current_best_ind[2],
                "epochs": current_best_ind[3]
            },
            "population": [
                {"chromosome": c, "loss": f, "accuracy": a, "composite": comp}
                for c, f, a, comp in zip(population, fitnesses, accuracies, composites)
            ]
        }) + "\n\n"

        # --- Next generation: selection, crossover, mutation ---
        next_population = []

        # Elitism: keep top 2 chromosomes unaffected by genetic operations
        if pop_size > 2:
            next_population.extend(population[:2])
        else:
            next_population.extend(population[:1])

        while len(next_population) < pop_size:
            # Tournament selection (size 2) — uses composites
            idx1, idx2 = random.sample(range(pop_size), 2)
            idx3, idx4 = random.sample(range(pop_size), 2)
            p1 = population[idx1] if composites[idx1] < composites[idx2] else population[idx2]
            p2 = population[idx3] if composites[idx3] < composites[idx4] else population[idx4]

            # Single-point crossover
            pt = random.randint(1, 3)
            child = p1[:pt] + p2[pt:]

            # Mutation logic
            if random.random() < mutation_rate:
                mut_idx = random.randint(0, 3)
                if mut_idx == 0:   child[0] = random.randint(10, 100)
                elif mut_idx == 1: child[1] = random.randint(1, 3)
                elif mut_idx == 2: child[2] = round(random.uniform(0.001, 0.05), 4)
                elif mut_idx == 3: child[3] = random.randint(10, 50)

            next_population.append(child)

        population = next_population

    # Stream completed flag signal
    yield json.dumps({
        "status": "completed",
        "best_overall_loss": best_overall_fitness_mse,
        "best_overall_accuracy": best_overall_accuracy,
        "best_overall_composite": best_overall_composite,
        "best_overall_chromosome": {
            "hidden_size": best_overall_chromosome[0],
            "num_layers": best_overall_chromosome[1],
            "learning_rate": best_overall_chromosome[2],
            "epochs": best_overall_chromosome[3]
        }
    }) + "\n\n"

    # Persist the best result to the database
    try:
        from .models import OptimizedHyperparams
        OptimizedHyperparams.objects.create(
            symbol               = symbol.upper(),
            hidden_size          = best_overall_chromosome[0],
            num_layers           = best_overall_chromosome[1],
            learning_rate        = best_overall_chromosome[2],
            epochs               = best_overall_chromosome[3],
            val_mse              = best_overall_fitness_mse,
            directional_accuracy = best_overall_accuracy,
            composite_fitness    = best_overall_composite
        )
    except Exception:
        pass  # never crash the stream on DB failure
