import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as matplotlib_colors
import matplotlib.pyplot as plt
import git

REPO_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir

SYMPTOM_NAMES = [
    "Fever",
    "Chills",
    "Cough",
    "Shortness of breath",
    "Sore throat",
    "Headache",
    "Fatigue",
    "Muscle weakness",
    "Anosmia",
    "Ageusia",
    "Nasal congestion",
    "Nausea",
    "Vomiting",
    "Diarrhea",
]


def plot_symptoms(symptom_df, symptoms="all", frame_type="RSV", scaled=True):
    """Plot symptom RSV OR Media curves. Save plot to plots/symptom

    Args:
        symptom_df: Pandas DataFrame with date column plus columns corresponding to symptom:<s> for s in symptoms
        symptoms:   Optional arg to specify specific set of symptoms. Defaults to all symptoms + combined
        frame_type: Choice of ['RSV', 'Media Count', 'Media Count Ratio']
        scaled:     Optional arg to min max scale data or not (scaled data is more visually distinguishable)

    """
    symptom_df["date"] = pd.to_datetime(symptom_df["date"])
    if symptoms == "all":
        symptoms = [symptom.lower() for symptom in SYMPTOM_NAMES]
        symptoms.append("combined")
    else:
        symptoms = [symptom.lower() for symptom in symptoms]
    NUM_COLORS = 20
    cm = plt.get_cmap("tab20")
    for i, symptom in enumerate(symptoms):
        if scaled:
            data = (
                symptom_df["symptom:" + symptom].to_numpy().reshape(-1, 1)
                / symptom_df["symptom:" + symptom].max()
            )
            data = symptom_df["symptom:" + symptom].to_numpy().reshape(-1, 1)
        else:
            data = symptom_df["symptom:" + symptom].to_numpy().reshape(-1, 1)
        plt.plot(
            symptom_df["date"], data, label=symptom, color=cm(i * 1.0 / NUM_COLORS)
        )
    plt.xlabel("Date")
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
    plt.xticks(rotation=45)
    plt.ylabel(f"Symptom {frame_type}")
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title(f"NY State 2021 Symptom {frame_type}")
    plt.savefig(
        f"{REPO_ROOT}/src/plots/symptom/ny_2021_symptom_{frame_type}.png",
        bbox_inches="tight",
    )
    plt.show()


def plot_individual_symptom_and_cases_mva(
    symptom_df, symptoms="all", frame_type="RSV", mv_window=5, use_moving_average=True
):
    """Plot symptom RSV OR Media vs daily new positive cases.
       Save plots to plots/time_varying_correlation/individual_symptom_vs_cases

    Args:
        symptom_df:         Pandas DataFrame with date column plus columns corresponding to symptom:<s> for s in symptoms
                            and daily_new_positives column for case counts
        symptoms:           Optional arg to specify specific set of symptoms. Defaults to all symptoms + combined
        frame_type:         Choice of ['RSV', 'Media Count', 'Media Count Ratio']
        mv_window:          Moving average window
        use_moving_average: Optional arg setting whether to use moving average or not. Set to False for raw symptom vs case count plots

    """
    symptom_df["date"] = pd.to_datetime(symptom_df["date"])

    if symptoms == "all":
        symptoms = [symptom.lower() for symptom in SYMPTOM_NAMES]
        symptoms.append("combined")

    for symptom in symptoms:
        fig, ax = plt.subplots()
        if not use_moving_average:
            symptom_col = "symptom:{}".format(symptom)
            ax.plot(
                symptom_df["date"],
                symptom_df[symptom_col].astype(float),
                color="tab:blue",
                label=symptom,
            )
        else:
            symptom_mv_col = "symptom:{}_mv({})".format(symptom, mv_window)
            ax.plot(
                symptom_df["date"],
                symptom_df[symptom_mv_col].astype(float),
                color="tab:blue",
                label="{} (mv = {})".format(symptom, mv_window),
            )
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_ylabel("{frame_type}")

        ax2 = ax.twinx()
        if not use_moving_average:
            case_col = "daily_new_positives"
            ax2.plot(
                symptom_df["date"],
                symptom_df[case_col].astype(float),
                color="tab:red",
                label="daily_new_positives",
            )
        else:
            case_mv_col = "daily_new_positives_mv({})".format(mv_window)
            ax2.plot(
                symptom_df["date"],
                symptom_df[case_mv_col].astype(float),
                color="tab:red",
                label="daily_new_positives (mv = {})".format(mv_window),
            )
        ax2.set_ylabel("Daily New Positives")

        # Plot legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2)

        if not use_moving_average:
            title = "{} {} vs Reported Cases".format(symptom.title(), frame_type)
            save_path = "{}/src/plots/time_varying_correlation/individual_symptom_vs_cases/ny_2021_{}_{}_vs_cases.png".format(
                REPO_ROOT, symptom, frame_type
            )
        else:
            title = "{} {} vs Reported Cases (MV = {})".format(
                symptom.title(), frame_type, mv_window
            )
            save_path = "{}/src/plots/time_varying_correlation/individual_symptom_vs_cases/ny_2021_{}_{}_vs_cases_mv({}).png".format(
                REPO_ROOT, symptom, frame_type, mv_window
            )
        plt.title(title)
        plt.savefig(
            save_path,
            bbox_inches="tight",
        )
        plt.show()


def plot_predictions_actual(
    results_df,
    model_name,
    rolling=False,
    rolling_window=7,
    ylabel="Case Count",
    set_title=True,
):
    """Plot curve of model predictions alongside actual case count curve. Save plot to plots/basic

    Args:
        results_df: Pandas DataFrame with columns ['date', 'predicted_case_count', 'actual_case_count']
        model_name: string name of model used

    """
    results_df["date"] = pd.to_datetime(results_df["date"])

    if rolling:
        results_df = pd.concat(
            [results_df["date"], results_df.rolling(rolling_window).mean()], axis=1
        )
        results_df["date"] = pd.to_datetime(results_df["date"])

    plt.plot(
        results_df["date"],
        results_df["predicted_case_count"].astype(float),
        label="predicted",
        color="red",
    )
    plt.plot(
        results_df["date"],
        results_df["actual_case_count"].astype(float),
        label="actual",
        color="green",
    )
    plt.xlabel("Date")
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    if set_title:
        plt.title(f"NY State 2021 {model_name} Predictions vs Actual Case Counts")
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(
        f"{REPO_ROOT}/src/plots/basic/ny_2021_{model_name}_predicted_vs_actual.png",
        bbox_inches="tight",
    )
    plt.show()


def plot_multiple_predictions_actual(
    results_df: list[pd.DataFrame],
    model_names: list[str],
    plot_name,
    rolling: bool = False,
    rolling_window: int = 7,
    ylabel: str = "Case Count",
    set_title: bool = True,
):
    """Plot curve of model predictions alongside actual case count curve. Save plot to plots/basic

    Args:
        results_df: List of Pandas DataFrame with columns ['date', 'predicted_case_count', 'actual_case_count']
        model_name: List string names of model used in the same order as results_df

    """
    assert (
        results_df and model_names
    ), "results_df and model_names must be non-empty lists."
    assert len(results_df) == len(
        model_names
    ), "results_df and model_names must be the same length."

    for i in range(len(results_df)):
        results_df[i] = results_df[i].reset_index(drop=True)

    for i in range(1, len(results_df)):
        assert (
            results_df[i]["date"] == results_df[0]["date"]
        ).all(), "All dataframes must have the same dates."

    # Custom color map without green-based colors
    custom_colors = [plt.cm.tab20(i) for i in range(20) if i != 4 and i != 5]
    custom_cmap = matplotlib_colors.ListedColormap(custom_colors)

    for i, (df, model_name) in enumerate(zip(results_df, model_names)):
        df["date"] = pd.to_datetime(df["date"])

        if rolling:
            df = pd.concat(
                [
                    df["date"],
                    df.rolling(rolling_window).mean(),
                ],
                axis=1,
            )
            df["date"] = pd.to_datetime(df["date"])

        plt.plot(
            df["date"],
            df["predicted_case_count"].astype(float),
            label=model_name,
            color=custom_cmap(i),
        )

    plt.plot(
        results_df[0]["date"],
        results_df[0]["actual_case_count"].astype(float),
        label="actual",
        color="green",
    )
    plt.xlabel("Date")
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    if set_title:
        plt.title(f"NY State 2021 {plot_name} Predictions vs Actual Case Counts")
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(
        f"{REPO_ROOT}/src/plots/basic/ny_2021_{plot_name}_predicted_vs_actual.png",
        bbox_inches="tight",
    )
    plt.show()


def plot_rolling_corr(
    symptom_df, symptoms="all", frame_type="RSV", rolling_corr_window=31, mv_window=5
):
    """Plot rolling correlation and rolling correlation of moving average of all passed in symptoms each on one plot.
       Save plots to plots/time_varying_correlation

    Args:
        symptom_df:             Pandas DataFrame with date column plus columns corresponding to symptom:<s>,
                                rolling_corr(<rolling_corr_window>):<s>, and
                                mv(<mv_window>)_rolling_corr({rolling_corr_window}):<s> for s in symptoms
        frame_type:             Choice of ['RSV', 'Media Count', 'Media Count Ratio']
        symptoms:               Optional arg to specify specific set of symptoms. Defaults to all symptoms + combined
        rolling_corr_window:    Rolling correlation window to use
        mv_window:              Moving average window to use

    """
    symptom_df["date"] = pd.to_datetime(symptom_df["date"])

    if symptoms == "all":
        symptoms = [symptom.lower() for symptom in SYMPTOM_NAMES]
        symptoms.append("combined")

    # Plot Rolling Correlation
    for symptom in symptoms:
        rolling_corr_col = "rolling_corr({}):{}".format(rolling_corr_window, symptom)
        plt.plot(
            symptom_df["date"],
            symptom_df[rolling_corr_col].astype(float),
            label=symptom,
        )
    plt.xlabel("Date")
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
    plt.xticks(rotation=45)
    plt.ylabel("Rolling Window Correlation")
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title(
        f"Rolling Window Correlation between Symptom {frame_type} and New Positives"
    )
    plt.savefig(
        f"{REPO_ROOT}/src/plots/time_varying_correlation/ny_2021_{frame_type}_rolling_corr({rolling_corr_window}).png",
        bbox_inches="tight",
    )
    plt.show()

    # Plot Rolling Correlation on Moving Average
    for symptom in symptoms:
        rolling_corr_col = "mv({})_rolling_corr({}):{}".format(
            mv_window, rolling_corr_window, symptom
        )
        plt.plot(
            symptom_df["date"],
            symptom_df[rolling_corr_col].astype(float),
            label=symptom,
        )
    plt.xlabel("Date")
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
    plt.xticks(rotation=45)
    plt.ylabel("Rolling Window Correlation")
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title(
        f"Rolling Window Correlation between Symptom {frame_type} and New Positives (MV = {mv_window})"
    )
    plt.savefig(
        f"{REPO_ROOT}/src/plots/ny_2021_{frame_type}_mv({mv_window})_rolling_corr({rolling_corr_window}).png",
        bbox_inches="tight",
    )
    plt.show()


def plot_individual_rolling_corr(
    symptom_df, symptoms="all", frame_type="RSV", rolling_corr_window=31, mv_window=5
):
    """Plot rolling correlation and rolling correlation of moving average of each of passed in symptoms on separate plot
       Save plots to plots/time_varying_correlation/individual_rolling_corr

    Args:
        symptom_df:             Pandas DataFrame with date column plus columns corresponding to symptom:<s>, rolling_corr(<rolling_corr_window>):<s>,
                                and mv(<mv_window>)_rolling_corr({rolling_corr_window}):<s> for s in symptoms
        symptoms:               Optional arg to specify specific set of symptoms. Defaults to all symptoms + combined
        frame_type:             Choice of ['RSV', 'Media Count', 'Media Count Ratio']
        rolling_corr_window:    Rolling correlation window to use
        mv_window:              Moving average window to use

    """
    symptom_df["date"] = pd.to_datetime(symptom_df["date"])

    if symptoms == "all":
        symptoms = [symptom.lower() for symptom in SYMPTOM_NAMES]
        symptoms.append("combined")

    for symptom in symptoms:
        rolling_corr_col = "rolling_corr({}):{}".format(rolling_corr_window, symptom)
        plt.plot(
            symptom_df["date"],
            symptom_df[rolling_corr_col].astype(float),
            label=symptom,
        )
        rolling_corr_col = "mv({})_rolling_corr({}):{}".format(
            mv_window, rolling_corr_window, symptom
        )
        plt.plot(
            symptom_df["date"],
            symptom_df[rolling_corr_col].astype(float),
            label="{} (mv = {})".format(symptom, mv_window),
        )

        plt.xlabel("Date")
        ax = plt.gca()
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
        plt.xticks(rotation=45)
        plt.ylabel("Rolling Window Correlation")
        plt.legend()
        plt.title(
            f"Rolling Window Correlation between {symptom.title()} {frame_type} and New Positives"
        )
        plt.savefig(
            f"{REPO_ROOT}/src/plots/time_varying_correlation/individual_rolling_corr/ny_2021_{symptom}_{frame_type}_rolling_corr({rolling_corr_window}).png",
            bbox_inches="tight",
        )
        plt.show()


## Keeping this around in case something goes wrong with setting use_moving_average flag on other method
# def plot_individual_symptoms_and_cases(symptom_df, symptoms='all', frame_type='RSV'):
#     """Plot symptom RSV OR Media vs daily new positive cases. Save plot to plots/symptom

#     Args:
#         symptom_df: Pandas DataFrame with date column plus columns corresponding to symptom:<s> for s in symptoms
#         symptoms:   Optional arg to specify specific set of symptoms. Defaults to all symptoms + combined
#         frame_type: Choice of ['RSV', 'Media Count', 'Media Count Ratio']

#     """
#     if symptoms == 'all':
#         symptoms = [symptom.lower() for symptom in SYMPTOM_NAMES]
#         symptoms.append('combined')

#     for symptom in symptoms:
#         fig, ax = plt.subplots()

#         symptom_col = "symptom:{}".format(symptom)
#         ax.plot(
#             symptom_df['date'],
#             symptom_df[symptom_col].astype(float),
#             color="tab:blue",
#             label=symptom
#         )

#         ax.set_xlabel("Date")
#         ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
#         ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
#         for tick in ax.get_xticklabels():
#             tick.set_rotation(45)
#         ax.set_ylabel(frame_type)

#         ax2 = ax.twinx()

#         case_col = "daily_new_positives"
#         ax2.plot(
#             symptom_df['date'],
#             symptom_df[case_col].astype(float),
#             color="tab:red",
#             label="daily_new_positives",
#         )
#         ax2.set_ylabel("Daily New Positives")

#         # Plot legend
#         h1, l1 = ax.get_legend_handles_labels()
#         h2, l2 = ax2.get_legend_handles_labels()
#         ax2.legend(h1 + h2, l1 + l2)

#         plt.xticks(rotation=45)
#         title = f"{symptom.title()} {frame_type} vs Reported Cases"
#         save_path = f"./plots/symptom/ny_2021_{symptom}_{frame_type}_vs_cases.png"
#         plt.title(title)
#         plt.savefig(
#             save_path,
#             bbox_inches="tight",
#         )
#         plt.show()
