import json
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from .ai_services import LSTMModel, get_price_history


# ============================================================================
# SECTION 1 — LSTM GENETIC ALGORITHM (unchanged from original)
# ============================================================================

def get_train_val_data(symbol="AAPL", seq_len=20):
    import ta
    df = get_price_history(symbol, period="2y")
    if df is None or df.empty:
        raise ValueError(f"Not enough data for {symbol}")

    df = df.resample('W').agg({
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum'
    }).dropna()

    close = df['Close']
    high  = df['High']
    low   = df['Low']

    df['rsi']       = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df['macd_diff'] = ta.trend.MACD(close=close).macd_diff()
    df['bb_width']  = (
        ta.volatility.BollingerBands(close=close, window=20).bollinger_hband() -
        ta.volatility.BollingerBands(close=close, window=20).bollinger_lband()
    )
    df['atr']      = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    sma20          = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df['sma_dist'] = (close - sma20) / sma20
    df['obv']      = ta.volume.OnBalanceVolumeIndicator(
        close=df['Close'], volume=df['Volume']
    ).on_balance_volume()

    feature_cols = ['Close', 'rsi', 'macd_diff', 'bb_width', 'atr', 'sma_dist', 'obv']
    df          = df[feature_cols].dropna()

    scaler      = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df.values)
    input_size  = data_scaled.shape[1]

    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i + seq_len])
        y.append(data_scaled[i + seq_len, 0])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    train_size       = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val,   y_val   = X[train_size:],  y[train_size:]

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val,   dtype=torch.float32),
        torch.tensor(y_val,   dtype=torch.float32),
        scaler,
        input_size
    )


def evaluate_chromosome(chromosome, symbol, input_size=None):
    hidden_size, num_layers, lr, epochs, dropout, seq_len = chromosome
    result     = get_train_val_data(symbol, seq_len=seq_len)
    input_size = result[5]
    X_train, y_train, X_val, y_val = result[:4]

    model     = LSTMModel(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss    = criterion(val_outputs, y_val)
        prev_prices      = X_val[:, -1, 0]
        actual_deltas    = y_val.squeeze()    - prev_prices
        predicted_deltas = val_outputs.squeeze() - prev_prices
        correct  = (torch.sign(actual_deltas) == torch.sign(predicted_deltas)).sum().item()
        accuracy = (correct / len(actual_deltas)) * 100 if len(actual_deltas) > 0 else 0.0

    composite = val_loss.item() * (1 - accuracy / 100)
    return val_loss.item(), accuracy, composite


def evolutionary_hpo_generator(symbol="AAPL", pop_size=5, generations=5, mutation_rate=0.2):
    """
    Generator: yields JSON lines of GA progress for LSTM HPO.
    Chromosome = [hidden_size, num_layers, lr, epochs, dropout, seq_len]
    """
    population = [[
        random.randint(10, 128), random.randint(1, 4),
        round(random.uniform(0.0005, 0.05), 4), random.randint(10, 60),
        round(random.uniform(0.0, 0.5), 2),     random.randint(10, 60)
    ] for _ in range(pop_size)]

    best_overall_composite  = float('inf')
    best_overall_mse        = None
    best_overall_accuracy   = None
    best_overall_chromosome = None

    for gen in range(1, generations + 1):
        adaptive_mutation = max(mutation_rate * (1 - (gen - 1) / generations), 0.05)
        fitnesses, accuracies, composites = [], [], []

        for ind in population:
            try:
                mse, acc, comp = evaluate_chromosome(ind, symbol)
            except Exception:
                mse, acc, comp = float('inf'), 0.0, float('inf')
            fitnesses.append(mse); accuracies.append(acc); composites.append(comp)

        idx        = np.argsort(composites)
        population = [population[i] for i in idx]
        fitnesses  = [fitnesses[i]  for i in idx]
        accuracies = [accuracies[i] for i in idx]
        composites = [composites[i] for i in idx]

        if composites[0] < best_overall_composite:
            best_overall_composite  = composites[0]
            best_overall_mse        = fitnesses[0]
            best_overall_accuracy   = accuracies[0]
            best_overall_chromosome = population[0]

        yield json.dumps({
            "generation":      gen,
            "best_loss":       fitnesses[0],
            "best_accuracy":   accuracies[0],
            "best_composite":  composites[0],
            "best_chromosome": {
                "hidden_size": population[0][0], "num_layers":    population[0][1],
                "learning_rate": population[0][2], "epochs":       population[0][3],
                "dropout":       population[0][4], "seq_len":      population[0][5]
            },
            "population": [
                {"chromosome": c, "loss": f, "accuracy": a, "composite": co}
                for c, f, a, co in zip(population, fitnesses, accuracies, composites)
            ]
        }) + "\n\n"

        # --- Next generation ---
        next_pop = population[:2] if pop_size > 2 else population[:1]
        while len(next_pop) < pop_size:
            i1, i2 = random.sample(range(pop_size), 2)
            i3, i4 = random.sample(range(pop_size), 2)
            p1 = population[i1] if composites[i1] < composites[i2] else population[i2]
            p2 = population[i3] if composites[i3] < composites[i4] else population[i4]
            pt    = random.randint(1, 5)
            child = p1[:pt] + p2[pt:]
            if random.random() < adaptive_mutation:
                m = random.randint(0, 5)
                if   m == 0: child[0] = random.randint(10, 128)
                elif m == 1: child[1] = random.randint(1, 4)
                elif m == 2: child[2] = round(random.uniform(0.0005, 0.05), 4)
                elif m == 3: child[3] = random.randint(10, 60)
                elif m == 4: child[4] = round(random.uniform(0.0, 0.5), 2)
                elif m == 5: child[5] = random.randint(10, 60)
            next_pop.append(child)
        population = next_pop

    yield json.dumps({
        "status":                   "completed",
        "best_overall_loss":        best_overall_mse,
        "best_overall_accuracy":    best_overall_accuracy,
        "best_overall_composite":   best_overall_composite,
        "best_overall_chromosome":  {
            "hidden_size":   best_overall_chromosome[0],
            "num_layers":    best_overall_chromosome[1],
            "learning_rate": best_overall_chromosome[2],
            "epochs":        best_overall_chromosome[3],
            "dropout":       best_overall_chromosome[4],
            "seq_len":       best_overall_chromosome[5],
        }
    }) + "\n\n"

    try:
        from .models import OptimizedHyperparams
        OptimizedHyperparams.objects.create(
            symbol               = symbol.upper(),
            hidden_size          = best_overall_chromosome[0],
            num_layers           = best_overall_chromosome[1],
            learning_rate        = best_overall_chromosome[2],
            epochs               = best_overall_chromosome[3],
            dropout              = best_overall_chromosome[4],
            seq_len              = best_overall_chromosome[5],
            val_mse              = best_overall_mse,
            directional_accuracy = best_overall_accuracy,
            composite_fitness    = best_overall_composite
        )
    except Exception:
        pass


# ============================================================================
# SECTION 2 — RL HYPERPARAMETER OPTIMISATION (Genetic Algorithm)
# ============================================================================
#
# Research basis:
#   Chromosome encodes the most impactful hyperparameters for each agent
#   based on sensitivity analysis in the FinRL and SB3 literature:
#
#   A2C gene vector: [learning_rate, n_steps, gamma, gae_lambda, ent_coef]
#   SAC gene vector: [learning_rate, batch_size, gamma, tau, init_ent_coef]
#   TD3 gene vector: [learning_rate, batch_size, gamma, tau, noise_sigma]
#
# Fitness function:
#   Quick training run (eval_timesteps=1500 steps) on the PortfolioEnv,
#   then measure the Sortino ratio of the evaluation episode.
#   fitness = -sortino  (we minimise, so higher Sortino = better agent)
#
# GA settings (kept small for speed):
#   pop_size=4, generations=3 per agent → 12 eval runs of 1500 steps each
#   This adds ~2-4 min overhead but meaningfully improves final performance.
# ============================================================================

# --- Gene search spaces ---
RL_GENE_SPACE = {
    "A2C": [
        ("learning_rate", "loguniform", 1e-5, 5e-4),
        ("n_steps",       "int",        64,   512),
        ("gamma",         "uniform",    0.95, 0.999),
        ("gae_lambda",    "uniform",    0.85, 1.0),
        ("ent_coef",      "loguniform", 1e-4, 0.05),
    ],
    "SAC": [
        ("learning_rate",  "loguniform", 1e-5, 5e-4),
        ("batch_size",     "int",        64,   256),
        ("gamma",          "uniform",    0.95, 0.999),
        ("tau",            "loguniform", 1e-3, 0.02),
        ("init_ent_coef",  "loguniform", 1e-4, 0.5),
    ],
    "TD3": [
        ("learning_rate", "loguniform", 1e-5, 5e-4),
        ("batch_size",    "int",        64,   256),
        ("gamma",         "uniform",    0.95, 0.999),
        ("tau",           "loguniform", 1e-3, 0.02),
        ("noise_sigma",   "uniform",    0.05, 0.3),
    ],
}


def _sample_gene(gene_spec):
    """Sample a single gene value from its search space."""
    name, dist, lo, hi = gene_spec
    if dist == "int":
        return random.randint(int(lo), int(hi))
    elif dist == "loguniform":
        return float(np.exp(random.uniform(np.log(lo), np.log(hi))))
    else:  # uniform
        return float(random.uniform(lo, hi))


def _mutate_gene(gene_spec, current_val, strength=1.0):
    """Perturb a gene value within its search space."""
    name, dist, lo, hi = gene_spec
    if dist == "int":
        delta = max(1, int((hi - lo) * 0.15 * strength))
        new   = int(np.clip(current_val + random.randint(-delta, delta), lo, hi))
        return new
    elif dist == "loguniform":
        log_val = np.log(current_val)
        log_range = np.log(hi) - np.log(lo)
        new = float(np.exp(np.clip(log_val + random.gauss(0, log_range * 0.2 * strength),
                                   np.log(lo), np.log(hi))))
        return new
    else:
        rng = (hi - lo)
        new = float(np.clip(current_val + random.gauss(0, rng * 0.2 * strength), lo, hi))
        return new


def _random_chromosome(agent_name):
    """Generate a random chromosome for the given agent."""
    return [_sample_gene(g) for g in RL_GENE_SPACE[agent_name]]


def _chromosome_to_hyperparams(agent_name, chromosome):
    """Convert a chromosome list to a named hyperparams dict."""
    names = [g[0] for g in RL_GENE_SPACE[agent_name]]
    return dict(zip(names, chromosome))


def _sortino(returns, rf_daily=0.05 / 252):
    """Sortino ratio from daily returns array."""
    if len(returns) < 5:
        return -999.0
    excess   = np.array(returns) - rf_daily
    downside = np.minimum(excess, 0.0)
    dd_std   = np.sqrt(np.mean(downside ** 2)) + 1e-8
    return float(np.mean(excess) / dd_std * np.sqrt(252))


def evaluate_rl_chromosome(chromosome, agent_name, price_data, ticker_features,
                            eval_timesteps=1500):
    """
    Train the specified agent for eval_timesteps steps using the chromosome
    hyperparameters, then evaluate one full episode and return:
        (fitness, sortino, final_return)

    fitness = -sortino  (lower = better, consistent with LSTM GA convention)
    """
    try:
        # Lazy imports — only needed when running HPO
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError:
            import gym
            from gym import spaces

        from stable_baselines3 import PPO, SAC, TD3, A2C
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.noise import NormalActionNoise

        # Import PortfolioEnv and cap helper from ai_services
        from .ai_services import PortfolioEnv, _cap_and_renormalize

        hp = _chromosome_to_hyperparams(agent_name, chromosome)

        n_assets = len(price_data.columns)

        # --- Build agent ---
        if agent_name == "A2C":
            env   = DummyVecEnv([lambda: PortfolioEnv(price_data, ticker_features)])
            model = A2C(
                "MlpPolicy", env, verbose=0,
                learning_rate = hp["learning_rate"],
                n_steps       = int(hp["n_steps"]),
                gamma         = hp["gamma"],
                gae_lambda    = hp["gae_lambda"],
                ent_coef      = hp["ent_coef"],
                policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
            )

        elif agent_name == "SAC":
            single_env = PortfolioEnv(price_data, ticker_features)
            model = SAC(
                "MlpPolicy", single_env, verbose=0,
                learning_rate = hp["learning_rate"],
                batch_size    = int(hp["batch_size"]),
                gamma         = hp["gamma"],
                tau           = hp["tau"],
                ent_coef      = hp["init_ent_coef"],
                policy_kwargs = dict(net_arch=[128, 128]),
            )

        else:  # TD3
            single_env   = PortfolioEnv(price_data, ticker_features)
            action_noise = NormalActionNoise(
                mean  = np.zeros(n_assets),
                sigma = hp["noise_sigma"] * np.ones(n_assets),
            )
            model = TD3(
                "MlpPolicy", single_env, verbose=0,
                learning_rate = hp["learning_rate"],
                batch_size    = int(hp["batch_size"]),
                gamma         = hp["gamma"],
                tau           = hp["tau"],
                action_noise  = action_noise,
                policy_kwargs = dict(net_arch=[128, 128]),
            )

        model.learn(total_timesteps=eval_timesteps)

        # --- Evaluate one full episode ---
        eval_env   = PortfolioEnv(price_data, ticker_features)
        obs, _     = eval_env.reset()
        done       = False
        while not done:
            action, _          = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = eval_env.step(action)

        pf = np.array(eval_env.portfolio_history)
        if len(pf) < 2:
            return 999.0, -999.0, 0.0

        daily_rets   = np.diff(pf) / (pf[:-1] + 1e-8)
        sortino      = _sortino(daily_rets)
        final_return = float((pf[-1] / pf[0]) - 1)
        fitness      = -sortino  # minimise → higher Sortino wins

        return fitness, sortino, final_return

    except Exception as e:
        return 999.0, -999.0, 0.0


def rl_hpo_for_agent(agent_name, price_data, ticker_features,
                     pop_size=4, generations=3, mutation_rate=0.3,
                     eval_timesteps=1500):
    """
    Compact GA for RL hyperparameter optimisation.

    Args:
        agent_name:      "A2C", "SAC", or "TD3"
        price_data:      aligned price DataFrame (from _fetch_data)
        ticker_features: dict of per-ticker signals (from fetch_ticker_features)
        pop_size:        4  (small — each eval takes ~10–30s)
        generations:     3  (3 × 4 = 12 eval runs total per agent)
        eval_timesteps:  1500 per fitness evaluation

    Returns:
        best_hyperparams: dict of evolved hyperparameter values
        best_sortino:     float — Sortino ratio achieved
        hpo_log:          list of dicts — generation-by-generation summary
    """
    gene_specs  = RL_GENE_SPACE[agent_name]
    population  = [_random_chromosome(agent_name) for _ in range(pop_size)]

    best_fitness    = float('inf')
    best_hyperparams = _chromosome_to_hyperparams(agent_name, population[0])
    best_sortino    = -999.0
    hpo_log         = []

    for gen in range(1, generations + 1):
        mutation_strength = 1.0 - (gen - 1) / generations  # decays each gen

        fitnesses = []
        sortinos  = []
        for chrom in population:
            fit, sor, _ = evaluate_rl_chromosome(
                chrom, agent_name, price_data, ticker_features, eval_timesteps
            )
            fitnesses.append(fit)
            sortinos.append(sor)

        # Sort by fitness (lower = better)
        sorted_idx = np.argsort(fitnesses)
        population = [population[i] for i in sorted_idx]
        fitnesses  = [fitnesses[i]  for i in sorted_idx]
        sortinos   = [sortinos[i]   for i in sorted_idx]

        if fitnesses[0] < best_fitness:
            best_fitness     = fitnesses[0]
            best_sortino     = sortinos[0]
            best_hyperparams = _chromosome_to_hyperparams(agent_name, population[0])

        hpo_log.append({
            "generation":      gen,
            "best_fitness":    round(fitnesses[0], 4),
            "best_sortino":    round(sortinos[0], 4),
            "best_hyperparams": best_hyperparams,
        })

        # Build next generation
        next_pop = population[:2] if pop_size > 2 else population[:1]
        while len(next_pop) < pop_size:
            # Tournament selection
            i1, i2 = random.sample(range(pop_size), 2)
            i3, i4 = random.sample(range(pop_size), 2)
            p1 = population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]
            p2 = population[i3] if fitnesses[i3] < fitnesses[i4] else population[i4]

            # Single-point crossover
            pt    = random.randint(1, len(gene_specs) - 1)
            child = p1[:pt] + p2[pt:]

            # Mutation
            if random.random() < mutation_rate:
                m_idx  = random.randint(0, len(gene_specs) - 1)
                child[m_idx] = _mutate_gene(gene_specs[m_idx], child[m_idx], mutation_strength)

            next_pop.append(child)

        population = next_pop

    return best_hyperparams, best_sortino, hpo_log


def run_rl_hpo_all_agents(price_data, ticker_features,
                          pop_size=4, generations=3, eval_timesteps=1500):
    """
    Run RL HPO for all three agents (A2C, SAC, TD3) and return a dict:
        {
            "A2C": {"hyperparams": {...}, "sortino": float, "log": [...]},
            "SAC": {...},
            "TD3": {...},
        }
    """
    results = {}
    for agent_name in ["A2C", "SAC", "TD3"]:
        try:
            best_hp, best_sor, log = rl_hpo_for_agent(
                agent_name, price_data, ticker_features,
                pop_size=pop_size, generations=generations,
                eval_timesteps=eval_timesteps,
            )
            results[agent_name] = {
                "hyperparams": best_hp,
                "sortino":     round(best_sor, 4),
                "log":         log,
            }
        except Exception as e:
            results[agent_name] = {
                "hyperparams": None,
                "sortino":     None,
                "log":         [{"error": str(e)}],
            }
    return results
