from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Tuple, TypeVar, Union

import polars as pl
from typing_extensions import Literal, ParamSpec

from functime.base.model import Model, ModelState
from functime.base.transformer import Transformer
from functime.ranges import make_future_ranges

# The parameters of the Model
P = ParamSpec("P")
# The return type of the esimator's curried function
R = Tuple[TypeVar("fit", bound=Callable), TypeVar("predict", bound=Callable)]

FORECAST_STRATEGIES = Optional[Literal["direct", "recursive", "naive"]]
DF_TYPE = Union[pl.LazyFrame, pl.DataFrame]


SUPPORTED_FREQ = [
    "1i",
    "1m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
    "1d",
    "1w",
    "1mo",
    "3mo",
    "1y",
]


@dataclass(frozen=True)
class ForecastState(ModelState):
    target: str
    target_schema: Mapping[str, pl.DataType]
    strategy: Optional[str] = "naive"
    features: Optional[List[str]] = None


class Forecaster(Model):
    """Autoregressive forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    target_transform : Optional[Transformer]
        functime transformer to apply to `y` before fit. The transform is inverted at predict time.
    feature_transform : Optional[Transformer]
        functime transformer to apply to `X` before fit and predict.
    **kwargs : Mapping[str, Any]
        Additional keyword arguments passed into underlying sklearn-compatible regressor.
    """

    def __init__(
        self,
        freq: Union[str, None],
        lags: int,
        max_horizons: Optional[int] = None,
        strategy: FORECAST_STRATEGIES = None,
        target_transform: Optional[Transformer] = None,
        feature_transform: Optional[Transformer] = None,
        **kwargs,
    ):

        if freq not in SUPPORTED_FREQ:
            raise ValueError(f"Offset {freq} not supported")

        self.freq = freq
        self.lags = lags
        self.max_horizons = max_horizons
        self.strategy = strategy
        self.target_transform = target_transform
        self.feature_transform = feature_transform
        self.kwargs = kwargs
        super().__init__()

    def __call__(
        self,
        y: DF_TYPE,
        fh: int,
        X: Optional[DF_TYPE] = None,
        X_future: Optional[DF_TYPE] = None,
    ) -> pl.DataFrame:
        self.fit(y=y, X=X)
        return self.predict(fh=fh, X=X_future)

    @property
    def name(self):
        return f"{self.__class__.__name__}(strategy={self.strategy})"

    def _transform_X(self, y: DF_TYPE, X: Optional[DF_TYPE] = None):
        # Feature transform
        if X is None:
            X_new = (
                y.select(y.columns[:2])
                .pipe(self.feature_transform)
                .collect(streaming=True)
                .lazy()
            )
        else:
            X_new = X.pipe(self.feature_transform).collect(streaming=True).lazy()
        return X_new

    def fit(self, y: DF_TYPE, X: Optional[DF_TYPE] = None):
        # Prepare y
        target_transform = self.target_transform
        y: pl.LazyFrame = self._set_string_cache(y.lazy().collect()).lazy()
        if target_transform is not None:
            y = y.pipe(target_transform).collect(streaming=True).lazy()
        # Prepare X
        if X is not None:
            if X.columns[0] == y.columns[0]:
                X = self._enforce_string_cache(X.lazy().collect())
            X = X.lazy()
        # Feature transform
        if self.feature_transform is not None:
            X = self._transform_X(X=X, y=y)
        # Fit AR forecaster
        artifacts = self._fit(y=y, X=X)
        # Prepare artifacts
        cutoffs = y.groupby(y.columns[0]).agg(pl.col(y.columns[1]).max().alias("low"))
        artifacts["__cutoffs"] = cutoffs.collect(streaming=True)
        state = ForecastState(
            entity=y.columns[0],
            time=y.columns[1],
            artifacts=artifacts,
            target=y.columns[-1],
            target_schema=y.schema,
            strategy=self.strategy or "recursive",
            features=X.columns[2:] if X is not None else None,
        )
        self.state = state
        self.target_transform = target_transform
        return self

    def predict(self, fh: int, X: Optional[DF_TYPE] = None) -> pl.DataFrame:

        from functime.forecasting._ar import predict_autoreg

        state = self.state
        entity_col = state.entity
        time_col = state.time
        target_col = state.target
        schema = state.target_schema
        # Cutoffs cannot be lazy
        cutoffs: pl.DataFrame = state.artifacts["__cutoffs"]
        # Get entity, time "index" for forecast
        future_ranges = make_future_ranges(
            time_col=state.time,
            cutoffs=cutoffs,
            fh=fh,
            freq=self.freq,
        )
        # Prepare X
        if X is not None:
            X = X.lazy()
            # Coerce X (can be panel / time series / cross sectional) into panel
            # and aggregate feature columns into lists
            has_entity = X.columns[0] == state.entity
            has_time = X.columns[1] == state.time

            if has_entity:
                X = self._enforce_string_cache(X.lazy().collect()).lazy()

            if has_entity and not has_time:
                X = future_ranges.lazy().join(X, on=entity_col, how="left")
            elif has_time and not has_entity:
                X = future_ranges.lazy().join(X, on=time_col, how="left")

        # Feature transform
        if self.feature_transform is not None:
            X = self._transform_X(
                X=X, y=future_ranges.explode(pl.all().exclude(entity_col))
            )

        y_pred_vals = predict_autoreg(self.state, fh=fh, X=X)
        # BUG: Exploding list[date] errogenously casts
        # into integer series but only for LazyFrame explode
        y_pred = (
            y_pred_vals.join(future_ranges, on=entity_col, how="left")
            .select([entity_col, time_col, target_col])
            .explode(pl.all().exclude(entity_col))
        )

        if self.target_transform is not None:
            y_pred = (
                y_pred.with_columns(pl.col(time_col).cast(schema[time_col]))
                .pipe(self.target_transform.invert)
                .collect(streaming=True)
            )

        y_pred = y_pred.pipe(self._reset_string_cache)
        return y_pred

    def backtest(
        self,
        y: DF_TYPE,
        X: Optional[pl.DataFrame] = None,
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        window_size: int = 10,
        strategy: Literal["expanding", "sliding"] = "expanding",
    ):
        from functime.backtesting import backtest
        from functime.cross_validation import (
            expanding_window_split,
            sliding_window_split,
        )

        if strategy == "expanding":
            cv = expanding_window_split(
                test_size=test_size,
                step_size=step_size,
                n_splits=n_splits,
            )
        else:
            cv = sliding_window_split(
                test_size=test_size,
                step_size=step_size,
                n_splits=n_splits,
                window_size=window_size,
            )
        y_preds, y_resids = backtest(
            forecaster=self,
            y=y,
            X=X,
            cv=cv,
            residualize=True,
        )
        return y_preds, y_resids

    def conformalize(
        self,
        fh: int,
        y: pl.DataFrame,
        X: Optional[DF_TYPE] = None,
        X_future: Optional[DF_TYPE] = None,
        alphas: Optional[List[float]] = None,
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        window_size: int = 10,
        strategy: Literal["expanding", "sliding"] = "expanding",
        return_results: bool = False,
    ) -> pl.DataFrame:
        from functime.conformal import conformalize

        y_pred = self.predict(fh=fh, X=X_future)
        y_preds, y_resids = self.backtest(
            y=y,
            X=X,
            test_size=test_size,
            step_size=step_size,
            n_splits=n_splits,
            window_size=window_size,
            strategy=strategy,
        )
        y_pred = pl.concat(
            [
                y_pred,
                y_preds.groupby(y_pred.columns[:2]).agg(
                    pl.col(y_pred.columns[-1])
                    .median()
                    .cast(y_pred.schema[y_pred.columns[-1]])
                ),
            ]
        )
        y_pred_qnts = conformalize(y_pred, y_resids, alphas=alphas)
        if return_results:
            return y_pred, y_pred_qnts, y_preds, y_resids
        return y_pred_qnts
