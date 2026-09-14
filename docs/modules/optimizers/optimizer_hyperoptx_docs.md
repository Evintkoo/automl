# HyperOptX Documentation

## Overview

The **HyperOptX** (Advanced Hyperparameter Optimization with Multi-Stage Optimization and Meta-Learning) class provides a sophisticated approach to hyperparameter optimization for machine learning models. It combines multiple advanced techniques including multi-fidelity optimization, surrogate modeling, Thompson sampling, and evolutionary strategies to efficiently search for optimal hyperparameter configurations.

HyperOptX is part of the AutoML optimization suite and distinguishes itself from simpler hyperparameter optimization techniques by providing:

1. **Multi-fidelity optimization** with adaptive resource allocation
2. **Meta-model selection** for surrogate model choice
3. **Thompson sampling** for exploration/exploitation balance
4. **Advanced acquisition functions** with entropy search
5. **Quasi-Monte Carlo methods** for efficient search space exploration
6. **Evolutionary strategies** for population-based training
7. **Constraint satisfaction** for parameter compatibility
8. **Transfer learning** from previous optimization runs
9. **Ensemble surrogate models** for improved prediction

## Features
- **Adaptive Budget Allocation**: Dynamically adjusts computational resources
- **Multi-Strategy Optimization**: Combines multiple optimization approaches
- **Meta-Learning**: Learns from previous optimization runs
- **Ensemble Methods**: Uses multiple surrogate models for robustness
- **Early Stopping**: Intelligent stopping criteria to prevent overfitting
- **Constraint Handling**: Supports complex parameter constraints

## Class: `HyperOptX`

### Initialization

```python
from modules.optimizer.hyperoptx import HyperOptX
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def __init__(
    self, 
    estimator,
    param_space,
    max_iter=100,
    cv=5,
    scoring=None,
    random_state=None,
    n_jobs=-1,
    verbose=0,
    maximize=True,
    time_budget=None,
    ensemble_surrogate=True,
    transfer_learning=True,
    optimization_strategy='auto',
    early_stopping=True,
    meta_learning=True,
    constraint_handling='auto'
):
    ...
```

### Parameters

- **estimator**: The machine learning estimator to optimize.
- **param_space**: Dictionary with parameter names as keys and search space as values:
  - For numerical parameters: tuple (low, high)
  - For categorical parameters: list of values
  - For distributions: scipy.stats distribution objects
- **max_iter** (default=100): Maximum number of iterations.
- **cv** (default=5): Number of cross-validation folds.
- **scoring** (default=None): Scoring function to use.
- **random_state** (default=None): Random state for reproducibility.
- **n_jobs** (default=-1): Number of jobs for parallel processing (-1 for all processors).
- **verbose** (default=0): Verbosity level.
- **maximize** (default=True): Whether to maximize or minimize the objective.
- **time_budget** (default=None): Maximum time budget in seconds.
- **ensemble_surrogate** (default=True): Whether to use ensemble surrogate models.
- **transfer_learning** (default=True): Whether to use transfer learning from previous runs.
- **optimization_strategy** (default='auto'): Strategy for optimization: 'auto', 'bayesian', 'evolutionary', 'hybrid'.
- **early_stopping** (default=True): Whether to use early stopping.
- **meta_learning** (default=True): Whether to use meta-learning for surrogate model selection.
- **constraint_handling** (default='auto'): Strategy for constraint handling: 'auto', 'penalty', 'projection', 'repair'.

### Attributes

- **best_params_**: The best hyperparameter set found during optimization.
- **best_score_**: The best score achieved during optimization.
- **best_estimator_**: The estimator fitted with the best hyperparameters.
- **cv_results_**: Dictionary containing detailed results from the optimization process:
  - `params`: List of evaluated parameter dictionaries
  - `mean_test_score`: Mean cross-validation scores
  - `std_test_score`: Standard deviations of cross-validation scores
  - `budget`: Resource budget used for each evaluation
  - `training_time`: Time taken for each evaluation
  - `iteration`: Iteration number for each evaluation
  - `surrogate_prediction`: Predictions from surrogate model
  - `surrogate_uncertainty`: Uncertainty estimates from surrogate model

### Public Methods

#### `fit(X, y) -> HyperOptX`

Runs the optimization process to find the best hyperparameters.

```python
def fit(self, X, y):
    ...
```

- **X**: Feature matrix (array-like or DataFrame) for training.
- **y**: Target vector (array-like).
- **Returns**: `self`, the fitted `HyperOptX` instance.

**Key steps** in `fit`:
1. **Initialization Phase**:
   - Analyzes parameter space
   - Initializes surrogate models, state tracking, and quasi-random sequence generators
2. **Phase 1 (Initial Exploration)**:
   - Samples a set of initial configurations
   - Evaluates them with reduced budget (50%)
   - Trains initial surrogate models
3. **Phase 2 (Iterative Optimization)**:
   - Uses selected optimization strategy (bayesian, evolutionary, or hybrid)
   - Proposes new configurations based on surrogate models and acquisition functions
   - Updates models and refines search progressively
   - Adapts budget allocation based on multi-fidelity schedule
4. **Phase 3 (Final Evaluation)**:
   - Re-evaluates top configurations with full budget
   - Selects and fits the best estimator

#### `score_cv_results() -> pandas.DataFrame`

Creates a DataFrame with all evaluation results and additional metrics.

```python
def score_cv_results(self):
    ...
```

- **Returns**: pandas.DataFrame with evaluation results.

#### `plot_optimization_history(figsize=(12, 8)) -> matplotlib.figure.Figure`

Visualizes the optimization process with four plots:
1. Score vs. iteration
2. Score vs. cumulative time
3. Parameter importance 
4. Surrogate model quality

```python
def plot_optimization_history(self, figsize=(12, 8)):
    ...
```

- **figsize** (default=(12, 8)): Size of the figure.
- **Returns**: matplotlib.figure.Figure object.

#### `benchmark_against_alternatives(X, y, methods=['grid', 'random', 'bayesian'], n_iter=50, cv=None, time_budget=None) -> dict`

Benchmarks HyperOptX against alternative hyperparameter optimization methods.

```python
def benchmark_against_alternatives(self, X, y, methods=['grid', 'random', 'bayesian'], 
                                n_iter=50, cv=None, time_budget=None):
    ...
```

- **X**: Feature matrix (array-like or DataFrame).
- **y**: Target vector (array-like).
- **methods** (default=['grid', 'random', 'bayesian']): Methods to benchmark against.
- **n_iter** (default=50): Number of iterations for each method.
- **cv** (default=None): Cross-validation folds (defaults to self.cv).
- **time_budget** (default=None): Time budget (in seconds) for each method.
- **Returns**: Dictionary with benchmark results.

### Key Internal Methods

#### Parameter Space Handling

- **`_analyze_param_space()`**: Determines parameter types (categorical, numerical, integer, distribution).
- **`_get_param_bounds()`**: Extracts bounds for numerical parameters.
- **`_analyze_param_constraints()`**: Analyzes parameter constraints and dependencies.
- **`_validate_params(params)`**: Validates and fixes parameter compatibility issues.

#### Configuration Encoding and Sampling

- **`_encode_config(config)`**: Encodes a configuration into a numerical vector with caching.
- **`_configs_to_features(configs)`**: Converts configurations to feature matrix.
- **`_decode_vector_to_config(vector, param_names)`**: Converts optimization vector back to parameter configuration.
- **`_sample_configurations(n, strategy='mixed')`**: Samples configurations using various strategies.

#### Surrogate Models and Acquisition

- **`_initialize_surrogate_models()`**: Initializes surrogate models based on meta-learning or ensemble strategy.
- **`_train_surrogate_models(X, y)`**: Trains surrogate models on evaluated configurations.
- **`_acquisition_function(x, models, best_f, xi=0.01)`**: Advanced acquisition function with ensemble support.
- **`_select_acquisition_function()`**: Selects acquisition function based on optimization stage.

#### Optimization Strategies

- **`_optimize_acquisition(surrogate_models, best_f, param_names, n_restarts=5)`**: Optimizes acquisition function.
- **`_optimize_continuous_space(surrogate_models, best_f, param_names)`**: Optimizes acquisition in continuous parameter space.
- **`_optimize_categorical_space(surrogate_models, best_f, param_names)`**: Optimizes acquisition in categorical parameter space.
- **`_optimize_mixed_space(surrogate_models, best_f, param_names)`**: Optimizes acquisition in mixed parameter space.
- **`_evolutionary_search(n_offspring=10, mutation_prob=0.2)`**: Performs evolutionary search based on current population.

#### Resource Allocation and Evaluation

- **`_multi_fidelity_schedule(max_iter)`**: Creates a multi-fidelity evaluation schedule.
- **`_successive_halving(configs, budget, n_survivors)`**: Evaluates configurations and keeps the best ones.
- **`_objective_func(params, budget=1.0, store=True)`**: Evaluates a configuration with the given budget.
- **`_needs_early_stopping(iteration, scores, times)`**: Determines if optimization should be stopped early.

## Typical Usage

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from hyperoptx import HyperOptX

# Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter space
param_space = {
    'n_estimators': (10, 200),      # Numerical parameter
    'max_depth': [None, 5, 10, 15, 20],  # Categorical parameter
    'min_samples_split': (2, 10),   # Numerical parameter
    'min_samples_leaf': (1, 5),     # Numerical parameter
    'bootstrap': [True, False]      # Categorical parameter
}

# Create and run the optimizer
optimizer = HyperOptX(
    estimator=RandomForestRegressor(),
    param_space=param_space,
    max_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1,
    maximize=True  # Negative MSE, so we maximize
)

optimizer.fit(X_train, y_train)

# Get results
print(f"Best parameters: {optimizer.best_params_}")
print(f"Best CV score: {optimizer.best_score_:.4f}")

# Evaluate on test set
y_pred = optimizer.best_estimator_.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {test_mse:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Visualize optimization history
fig = optimizer.plot_optimization_history()

# Compare with other methods
results = optimizer.benchmark_against_alternatives(
    X_train, y_train,
    methods=['grid', 'random', 'bayesian'],
    n_iter=30,
    time_budget=60  # 1 minute timeout
)
```

## Advanced Features

### Multi-Fidelity Optimization

HyperOptX implements a multi-fidelity approach to hyperparameter optimization. It starts evaluating configurations with a smaller budget (e.g., fewer cross-validation folds or a subset of the data) and progressively increases the resource allocation for promising configurations. This approach allows exploring more configurations with limited computational resources.

```python
# Budget schedule increases from 20% to 100% over iterations
budget_schedule = self._multi_fidelity_schedule(self.max_iter)
```

### Meta-Learning and Transfer Learning

The meta-learning component tracks the performance of different surrogate models across iterations and problem types. This information helps select the most appropriate surrogate model for the current optimization task. Transfer learning can leverage knowledge from previous optimization runs to warm-start new optimization tasks.

```python
# Record for meta-learning
self.meta_learning_data['problem_features'].append({
    'n_samples': X.shape[0],
    'n_features': X.shape[1],
    'y_mean': np.mean(y_valid),
    'y_std': np.std(y_valid),
    'iteration': self.iteration_count
})
self.meta_learning_data['best_surrogate'].append(current_best)
self.meta_learning_data['surrogate_performance'].append(model_errors)
```

### Ensemble Surrogate Models

HyperOptX can combine multiple surrogate models (Gaussian Process, Random Forest, Neural Network) to improve prediction accuracy and uncertainty estimates. The weights of different models are dynamically adjusted based on their performance.

```python
# Weighted mean and variance for ensemble models
if self.surrogate_weights is not None:
    weights = self.surrogate_weights
    mu = np.zeros_like(all_means[0])
    
    # Compute weighted mean
    for i, mean in enumerate(all_means):
        mu += weights[i] * mean
        
    # Compute weighted variance (including model disagreement)
    total_var = np.zeros_like(all_stds[0])
    
    # Within-model variance
    for i, std in enumerate(all_stds):
        total_var += weights[i] * (std ** 2)
        
    # Between-model variance (disagreement)
    for i, mean in enumerate(all_means):
        total_var += weights[i] * ((mean - mu) ** 2)
        
    sigma = np.sqrt(total_var)
```

### Hybrid Optimization Strategies

HyperOptX supports multiple optimization strategies and can dynamically select the most appropriate one based on the problem characteristics:

- **Bayesian**: Uses surrogate models and acquisition functions
- **Evolutionary**: Uses genetic algorithms with selection, crossover, and mutation
- **Hybrid**: Combines both approaches

```python
# Strategy determination
selected_strategy = self.optimization_strategy
if selected_strategy == 'auto':
    # Choose strategy based on problem size
    if X.shape[1] > 50 or len(self.param_space) > 10:
        # High-dimensional problems: prefer evolutionary
        selected_strategy = 'evolutionary'
    elif len([p for p, t in self.param_types.items() if t == 'categorical']) > len(self.param_types) / 2:
        # Many categorical parameters: prefer bayesian
        selected_strategy = 'bayesian'
    else:
        # Default to hybrid
        selected_strategy = 'hybrid'
```

## Notes and Best Practices

- **Problem Types**: HyperOptX is designed to work with any scikit-learn compatible estimator, including classifiers, regressors, and pipelines.

- **Parameter Space Definition**: The quality of optimization depends on well-defined parameter spaces. Use domain knowledge to set reasonable bounds and include important parameters.

- **Resource Management**: For computationally expensive models, consider using:
  - Lower `max_iter` values (30-50 can be sufficient)
  - `time_budget` to limit total optimization time
  - `early_stopping=True` to terminate optimization when progress plateaus

- **Parallelization**: Set `n_jobs=-1` to use all available CPU cores, but be mindful of memory usage for large datasets.

- **Optimization Strategies**:
  - `'bayesian'`: Best for small to medium parameter spaces with expensive evaluations
  - `'evolutionary'`: Better for high-dimensional spaces or many categorical parameters
  - `'hybrid'`: Good default that balances exploration and exploitation

- **Visualization**: The `plot_optimization_history()` method provides valuable insights into the optimization process and parameter importance.

- **Benchmarking**: Use `benchmark_against_alternatives()` to compare HyperOptX against traditional methods like GridSearchCV and RandomizedSearchCV.

## Summary

HyperOptX provides a comprehensive solution for hyperparameter optimization, combining multiple advanced techniques to efficiently find optimal configurations. Its adaptive strategies, surrogate modeling, and multi-fidelity approach make it particularly effective for complex models with large parameter spaces and computationally expensive evaluations.