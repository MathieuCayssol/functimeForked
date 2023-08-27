import polars as pl

def benford_correlation(x: pl.Series)-> pl.Expr:
    X = (
        (x / (10 ** x.abs().log10().floor()))
        .abs()
        .floor()
    )
    df_corr = pl.DataFrame(
        [
            [X.eq(i).mean() for i in pl.int_range(1, 10, eager=True)],
            (1+1/pl.int_range(1, 10, eager=True)).log10()
        ]
    ).corr()
    return df_corr[0,1]

def _get_length_sequences_where(x: pl.Series)-> pl.Series:
    X = (
        x
        .alias("orig")
        .to_frame()
        .with_columns(
            shift=pl.col("orig").shift(periods=1)
        )
        .with_columns(
            mask=pl.col("orig").ne(pl.col("shift")).fill_null(0).cumsum()
        )
        .filter(pl.col("orig") == 1)
        .groupby(pl.col("mask")).count()
    )["count"]
    return X

def longest_strike_below_mean(x: pl.Expr)-> pl.Expr:
    X = _get_length_sequences_where(x = (x < x.mean()))
    return X.max() if X.len() > 0 else 0


def longest_strike_above_mean(x: pl.Expr)-> pl.Expr:
    X = _get_length_sequences_where(x = (x > x.mean()))
    return X.max() if X.len() > 0 else 0

def mean_n_absolute_max(x: pl.Expr, n_maxima: int)-> pl.Expr:
    return x.abs().sort(descending=True)[:n_maxima].mean() if x.len() > n_maxima else None

def percent_reocurring_points(x: pl.Expr):
    X = x.value_counts().filter(
        pl.col("counts") > 1
    ).sum()
    return X[0, "counts"] / x.len()

def percent_recoccuring_values(x: pl.Expr):
    X = x.value_counts().filter(
        pl.col("counts") > 1
    )
    return X.shape[0] / x.n_unique()

def sum_reocurring_points(x: pl.Expr):
    X = x.value_counts().filter(
        pl.col("counts") > 1
    )
    return X[:,0].dot(X[:,1])

def sum_reocurring_values(x: pl.Expr):
    X = x.value_counts().filter(
        pl.col("counts") > 1
    ).sum()
    return X[0,0]