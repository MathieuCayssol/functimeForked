# functime

![functime](img/banner.png)

## Production-ready time series models

**functime** is a machine learning library for time-series predictions that just works.

- **Fast:** Forecast 100,000 time series in seconds *on your laptop*
- **Cloud-native:** Instantly run, deploy, and serve predictive models
- **Efficient:** Embarressingly parallel feature engineering using [**Polars**](https://www.pola.rs/) *
- **Battle-tested:** Algorithms that deliver real business impact and win competitions
- **Scalable-by-default:** Every model is designed to fit and predict / transform [collections of time series](#global-forecasting)

## Additional Highlights

- Every forecaster supports **exogenous features**
- Utilities to add calendar effects, special events (e.g. holidays), weather patterns (coming soon), and economic trends (coming soon)
- **Backtesting** with expanding window and sliding window splitters
- Automated lags and **hyperparameter tuning** using [`FLAML`](https://github.com/microsoft/FLAML)
- **Probablistic forecasts** via quantile regression and conformal prediction
- Supports recursive and direct forecast strategies

## Installation

Check out this [guide](installation.md) to install functime. Requires Python 3.8+.

## Example Usage

??? tip "Show me the code: `quickstart.py`"

    Want to go straight into code?
    Run through every example in "Quick Start" with the following script:

    ```python
    import polars as pl

    from functime.cross_validation import train_test_split
    from functime.feature_extraction import (
        add_calendar_effects,
        add_holiday_effects
    )

    # Load data
    y = pl.read_parquet("https://bit.ly/commodities-data")
    entity_col, time_col = y.columns[:2]
    X = (
        y.select([entity_col, time_col])
        .pipe(add_calendar_effects(["weekday", "month", "year"]))
        .pipe(add_holiday_effects(country_codes=["US"], freq="1d"))
        .collect()
    )

    print("🎯 Target variable (y):\n", y)
    print("📉 Exogenous variables (X):\n", X)

    # Train-test splits
    test_size = 3
    freq = "1mo"
    y_train, y_test = train_test_split(test_size)(y)
    X_train, X_test = train_test_split(test_size)(X)
    ```

## Forecasting

Load a collection of time series, also known as panel data, into [`polars.LazyFrames`](https://pola-rs.github.io/polars/py-polars/html/reference/lazyframe/index.html) (recommended) or `polars.DataFrames` and split them into train/test subsets.

```python
from functime.forecasting import LinearModel
from functime.metrics import mase

# Fit
forecaster = LinearModel(lags=12, freq="1mo")
forecaster.fit(y=y_train)

# Predict
y_pred = forecaster.predict(fh=3)

# Score
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```

!!! info "Supported Data Schemas"

    `X: polars.LazyFrame | polars.DataFrame` and `y: polars.LazyFrame | polars.DataFrame` must contain at least three columns.
    The first column must represent the `entity` / `series_id` dimension.
    The second column must represent the `time` dimension as an integer, `pl.Date`, or `pl.Datetime` series.
    Remaining columns are considered as features.

!!! tip "functime ❤️ currying"

    Every `transformer` and `splitter` are [curried functions](https://composingprograms.com/pages/16-higher-order-functions.html#currying).

    ```python
    from functime.preprocessing import boxcox, impute
    from functime.cross_validation import expanding_window_split

    # Use df.pipe to chain operations together
    X_splits: pl.LazyFrame = (
        X.pipe(boxcox(method="mle"))
        .pipe(impute(method="linear"))
        .pipe(expanding_window_split(test_size=28, n_splits=3, step_size=1))
    )
    # Call .collect to execute query
    X_splits = X_splits.collect()
    ```

    You can also use any `forecaster` as a curried function to run fit-predict in a single line of code.

    ```python
    from functime.forecasting import LinearModel

    y_pred = LinearModel(lags=12, freq="1mo")(
        y=y_train,
        fh=28,
        X=X_train,
        X_future=X_test
    )
    ```

??? warning "functime is lazy"

    `transformers` and `splitters` in `cross_validation`, `feature_extraction`, and `preprocessing` are lazy.
    These callables return `LazyFrames`, which represents a Lazy computation graph/query against the input `DataFrame` / `LazyFrame`.
    **No computation is run until the `collect()` method is called on the `LazyFrame`.**

    `X` and `y` should be preprocessed lazily for optimal performance.
    Lazy evaluation allows `polars` to optimize all operations on the input `DataFrame` / `LazyFrame` at once.
    Lazy preprocessing in `functime` allows for more efficient `groupby` operations.

    With lazy transforms, operations series-by-series (e.g. `boxcox`, `impute`, `diff`) are chained in parallel: `groupby` is only called once.
    By contrast, with eager transforms, operations series-by-series is called in sequence: `groupby-aggregate` is called per transform.

### Global Forecasting

Every `forecaster` exposes a scikit-learn `fit` and `predict` API.
The `fit` method takes `y` and `X` (optional).
The `predict` method takes the forecast horizon `fh: int`, frequency alias `freq: str`, and `X` (optional).

??? info "Supported Frequency Aliases"

    - 1ns (1 nanosecond)
    - 1us (1 microsecond)
    - 1ms (1 millisecond)
    - 1s (1 second)
    - 1m (1 minute)
    - 1h (1 hour)
    - 1d (1 day)
    - 1w (1 week)
    - 1mo (1 calendar month)
    - 1y (1 calendar year)
    - 1i (1 index count)


```python
from functime.forecasting import LinearModel
from functime.metrics import mase

# Fit
forecaster = LinearModel(lags=12, freq="1mo")
forecaster.fit(y=y_train)

# Predict
y_pred = forecaster.predict(fh=3)

# Score
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```

!!! question "Global vs Local Forecasting"

    **`functime` only supports global forecasters.**
    Global forecasters fit and predict a collection of time series using a single model.
    Local forecasters (e.g. ARIMA, ETS, Theta) fit and predict one series per model.
    Example collections of time series, which are also known as panel data, include:

    - Sales across product in a retail store
    - Churn rates across customer segments
    - Sensor data across devices in a factory
    - Delivery times across trucks in a logistics fleet

    Global forecasters, trained on a collection of similar time series, consistently outperform local forecasters.[^1]
    Most notably, all top 50 competitors in the M5 Forecasting Competition used a global LightGBM forecasting model.[^2]

    [^1]: Montero-Manso, P., & Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of time series: Locality and globality. International Journal of Forecasting, 37(4), 1632-1653.

    [^2]: Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting.

!!! tip "Save 100x in Cloud spend"

    Local forecasting is expensive and slow.
    To productionize forecasts at scale (>1,000 series), local models have no choice but distributed computing.
    Every fit-predict call per local model per series are executed in parallel across the distributed cluster.
    Running a distributed cluster, however, is a significant cost and time sink for any data team.

    **`functime` believes that the vast majority of businesses do not need distributed computing to produce high-quality forecasts**.
    Every `forecaster`, `transformer`, `splitter`, and `metric` in `functime` operates globally across collections of time series.
    We rewrote every time series operation in `polars` for blazing fast multi-threaded parallelism.

    Stop using Databricks to scale your forecasts. Use `functime`.

### Exogenous Features

Every `forecaster` supports exogenous regressors.

```python
from functime.forecasting import LinearModel

forecaster = LinearModel(lags=12, fit_intercept=False, freq="1mo")
forecaster.fit(y=y_train, X=X_train)
y_pred = forecaster.predict(fh=3, X=X_test)
```

### Forecast Strategies

`functime` supports three forecast strategies: `recursive`, `direct` multi-step, and a simple ensemble of both `recursive` and `direct`.

```python
from functime.forecasting import LinearModel

# Recursive (Default)
recursive_model = LinearModel(strategy="recursive")
y_pred_rec = recursive_model(y_train, fh)

# Direct
max_horizons = 12  # Number of direct models
direct_model = = LinearModel(strategy="direct",max_horizons=max_horizons, freq="1mo")
y_pred_dir = recursive_model(y_train, fh)

# Ensemble
ensemble_model = LinearModel(strategy="ensemble", max_horizons=max_horizons, freq="1mo")
y_pred_ens = ensemble_model(y=y_train, fh=3)
```
where `max_horizons` is the number of models specific to each forecast horizon.
For example, if `max_horizons = 12`, then twelve forecasters are fitted in total: the 1-step ahead forecast, the 2-steps ahead forecast, the 3-steps ahead forecast, ..., and the final 12-steps ahead forecast.

### AutoML

#### Optimal Lag Length

`auto_{model}` forecasters automatically select the optimal number of lags via cross-validation.
These forecasters conduct a search over possible models within `min_lags` and `max_lags`.
The best model is the model with the lowest average RMSE (root mean squared error) across splits.

```python
from functime.forecasting import AutoLinearModel

# Fit then predict
forecaster = AutoLinearModel(min_lags=3, max_lags=6, freq="1mo")
forecaster.fit(y=y_train, X=X_train)
y_pred = forecaster.predict(fh=3, X=X_test)

# Fit and predict
y_pred = AutoLinearModel(min_lags=3, max_lags=6, freq="1mo")(
    y=y_train,
    X=X_train,
    X_future=X_test,
    fh=3
)
```

#### Hyperparameter Tuning
`auto_{model}` forecasters automatically select the optimal number of lags via cross-validation.
These forecasters conduct a search over possible models within `min_lags` and `max_lags`.
The best model is the model with the lowest average RMSE (root mean squared error) across splits.

`functime` uses [`FLAML`](https://microsoft.github.io/FLAML/docs/getting-started) under the hood to conduct hyperparameter tuning.

```python
from flaml import tune
from functime.forecasting import AutoLightGBM

# Specify search space, initial conditions, and time budget
search_space = {
    "reg_alpha": tune.loguniform(1e-08, 10.0),
    "reg_lambda": tune.loguniform(1e-08, 10.0),
    "num_leaves": tune.randint(
        2, 2**max_depth if max_depth > 0 else 2**DEFAULT_TREE_DEPTH
    ),
    "colsample_bytree": tune.uniform(0.4, 1.0),
    "subsample": tune.uniform(0.4, 1.0),
    "subsample_freq": tune.randint(1, 7),
    "min_child_samples": tune.qlograndint(5, 100, 5),
}
points_to_evaluate = [
    {
        "num_leaves": 31,
        "colsample_bytree": 1.0,
        "subsample": 1.0,
        "min_child_samples": 20,
    }
]
time_budget = 420

# Fit model
forecaster = AutoLightGBM(
    freq="1mo',
    min_lags=3,
    max_lags=6,
    time_budget=time_budget,
    search_space=search_space,
    points_to_evaluate=points_to_evaluate
)
forecaster.fit(y=y_train)

# Get best lags and model hyperparameters
best_params = forecast.state.artifacts["best_params"]
```

!!! tip "Sane Hyperparameter Defaults"

    Sane defaults are used if `search_space` or `points_to_evaluate` are left as `None`.
    `functime` specify default hyperparameters search spaces according to best-practices from industry, top Kaggle solutions, and research.


### Backtesting

#### Expanding Window
View reference for [`expanding_window_split`](https://docs.functime.ai/ref/cross-validation/#functime.cross_validation.expanding_window_split).

#### Sliding Window
View reference for [`sliding_window_split`](https://docs.functime.ai/ref/cross-validation/#functime.cross_validation.sliding_window_split).

#### Backtest Function

#### Best Practices

### Probablistic Forecasts

!!! success "Quantile and Conformal Prediction Intervals"

    `functime` supports two methods for generating prediction intervals:

    === "Quantile"

        ```python
        from functime.forecasting import AutoLightGBM

        # Forecasts at 10th and 90th percentile
        y_pred_10 = AutoLightGBM(alpha=0.1, freq="1d")(y=y_train, fh=28)
        y_pred_90 = AutoLightGBM(alpha=0.9, freq="1d")(y=y_train, fh=28)
        ```

    === "Conformal"

        Only ensemble batch prediction intervals (EnbPI) from the paper [Conformal prediction interval for dynamic time-series](https://arxiv.org/abs/2010.09107) are currently supported.

        ```python
        from functime.conformal import conformalize

        # First run backtest with `residualize` set to `True`
        y_preds, y_conformal_scores = backtest(
            forecaster,
            y=y_train,
            X=X_train,
            fh=3,
            freq="1mo",
            step_size=1,
            n_splits=3
        )

        # Forecasts at 10th and 90th percentile
        y_pred_quantiles = conformalize(y_pred, y_resids, alphas=[0.1, 0.9])
        ```

### Future Direction

Follow us on [LinkedIn](https://www.linkedin.com/company/functime/) to receive the latest updates.
