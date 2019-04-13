import pandas as pd
import numpy as np
import scikits.bootstrap as sci


def time_across_length(runs, ids, times, sequence_lengths):
    df_min = pd.DataFrame({"run": runs, "time": times}, index=ids)
    df_min = df_min.pivot(index=df_min.index, columns="run")["time"]
    df_min = df_min.min(axis=1)
    df_min = pd.DataFrame(df_min, columns=["time"])

    df_lengths = pd.DataFrame.from_dict(sequence_lengths, orient="index")
    df_lengths.columns = ["length"]  # pandas 0.22 from-dict does not allow doing it dire.

    def fill_not_solved(df):
        for count in range(1, len(df_lengths)):
            df = df.fillna(5000 + 500 * count ** 1.6, limit=1)
        return df

    length_grouped = df_lengths.join(df_min, how="outer").groupby("length")
    length_grouped = length_grouped.transform(fill_not_solved)

    df = df_lengths.join(length_grouped, how="outer")
    df = df.set_index("length")
    return df.sort_index()


def _add_timeout_row(df, timeout):
    last_row = df.tail(1)
    last_row.index = [timeout]
    return df.append(last_row)


def _add_start_row(df):
    start_row = pd.DataFrame({column: [0] for column in df.columns}, index=[1e-10])
    return start_row.append(df)


def solved_across_time_per_run(runs, times, ci_alpha, dataset_size, timeout):
    # 1.
    df = pd.DataFrame({"run": runs, "times": times})
    df = df.sort_values("times")

    # 2.
    # Duplicates make inverting with pivot impossible so count duplicates and add in 3
    # From https://stackoverflow.com/a/41269427
    df = (
        df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: "counts"})
    )
    df = df.set_index("times")
    df = df.pivot(index=df.index, columns="run")["counts"]

    # 3.
    df = df.fillna(0)
    df = df.apply(np.cumsum)

    # 4.
    mean = df.mean(axis=1)
    mean.name = "mean"
    low_ci, high_ci = sci.ci(
        df.T, alpha=ci_alpha, statfunction=lambda x: np.average(x, axis=0)
    )
    low_ci = pd.Series(low_ci, index=df.index, name=f"low_ci_{ci_alpha}")
    high_ci = pd.Series(high_ci, index=df.index, name=f"high_ci_{ci_alpha}")

    # print(df.std(axis=1) / np.sqrt(df.shape[1]))
    # print(df)
    # print(dataset_size)

    low_ci = mean - df.std(axis=1)  # / np.sqrt(df.shape[1])
    low_ci.name = f"low_ci_{ci_alpha}"

    # print(df.shape[1])
    high_ci = mean + df.std(axis=1)  # / np.sqrt(df.shape[1])
    high_ci.name = f"high_ci_{ci_alpha}"


    df = pd.concat([mean, low_ci, high_ci], axis=1) / dataset_size * 100
    return _add_start_row(_add_timeout_row(df, timeout))


def solved_across_time_min(runs, ids, times, dataset_size, timeout):
    df = pd.DataFrame({"run": runs, "times": times}, index=ids)
    df = df.pivot(index=df.index, columns="run")["times"]
    df = df.min(axis=1)
    df = df.sort_values()

    df = pd.DataFrame({"solved": list(range(1, len(df) + 1))}, index=df)
    df = df[~df.index.duplicated(keep="last")] / dataset_size * 100

    return _add_start_row(_add_timeout_row(df, timeout))


def runs_solve_instance(runs, ids):
    df = pd.DataFrame({"id": ids, "runs": runs})
    return df.groupby("id").count()


def _solved_till_time_limit(df, time_limit):
    return df[df["time"] < time_limit].groupby("id").id.count().size


def solved_per_time_limit(
    runs, ids, times, time_limits=(60, 300, 600, 1200, 3600, 7200, 14400, 36000)
):
    df = pd.DataFrame({"id": ids, "run": runs, "time": times})
    counts_per_limit = [_solved_till_time_limit(df, time) for time in time_limits]
    return pd.DataFrame({"time": time_limits, "solved": counts_per_limit}).set_index(
        "time"
    )


def _per_quantile(df, quantile):
    return df[df["runs"] >= quantile].groupby("id").id.count().size


def solved_per_run_quantile(
    runs, ids, attempts, quantiles=(1, 5, 10, 25, 50, 75, 90, 95, 100)
):
    df = runs_solve_instance(runs, ids).reset_index()
    df["runs"] = (df["runs"] / attempts) * 100
    solved_per_quantile = [_per_quantile(df, quantile) for quantile in quantiles]
    return pd.DataFrame({"quantile": quantiles, "solved": solved_per_quantile}).set_index(
        "quantile"
    )
