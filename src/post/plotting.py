import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.post.consts import PALETTE
from src.post.report import Report


def render_as_1_figure(report: Report, font_size: int = 11):
    # Set global font size for all plots
    plt.rcParams.update({"font.size": font_size})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), constrained_layout=True)
    plot_accuracy(report.final_df, ax=ax1)
    plot_execution_time_distribution(report.original_df, ax=ax2)
    plt.show()


def render_as_multiple_figures(report: Report, font_size: int = 11):
    # Set global font size for all plots
    plt.rcParams.update({"font.size": font_size})

    fig1, ax1 = plt.subplots(figsize=(15, 6), constrained_layout=True)
    plot_accuracy(report.final_df, ax=ax1)
    fig2, ax2 = plt.subplots(figsize=(15, 6), constrained_layout=True)
    plot_execution_time_distribution(report.original_df, ax=ax2)
    plt.show()


def plot_accuracy(final_df, ax):
    specific_model_temp = "QWEN / T=0.2 / GPU"
    sorted_prompts = (
        final_df[final_df["Configuration"] == specific_model_temp]
        .sort_values(by="Accuracy", ascending=False)["Prompt Template"]
        .unique()
    )

    final_df["Prompt Template"] = pd.Categorical(
        final_df["Prompt Template"], categories=sorted_prompts, ordered=True
    )

    sns.barplot(
        data=final_df,
        x="Prompt Template",
        y="Accuracy",
        hue="Configuration",
        palette=PALETTE,
        ax=ax,
        order=sorted_prompts,
    )

    rotate_xticks(ax, sorted_prompts, rotation=45, ha="right")

    ax.yaxis.grid(True, linestyle=":", linewidth=0.7)
    ax.set_title("Accuracy of Each Prompt Template for Different Configurations")
    ax.set_xlabel("Prompt Template")
    ax.set_ylabel("Accuracy")

    ax.legend(
        title="Configuration",
        loc="lower center",
        ncol=4,
    )


def plot_execution_time_distribution(original_df, ax):
    original_df["Experiment"] = (
        original_df["Configuration"] + ",\n" + original_df["Prompt Template"]
    )

    experiment_medians = (
        original_df.groupby("Experiment")["Execution Time (minutes)"]
        .median()
        .sort_values()
    )

    sorted_df = (
        original_df.set_index("Experiment").loc[experiment_medians.index].reset_index()
    )

    sns.boxplot(
        data=sorted_df,
        x="Experiment",
        y="Execution Time (minutes)",
        hue="Configuration",
        palette=PALETTE,
        ax=ax,
        dodge=False,
    )

    # Extract only the part after the newline character for each label
    # Why: The part before the newline is the 'Configuration', which is already
    # represented by the hue in the plot. To avoid redundancy and keep the x-tick
    # labels focused on the unique part, we strip out the 'Configuration' and keep
    # only the 'Prompt Template', which provides additional relevant information.
    experiment_labels = sorted_df["Experiment"].unique()
    experiment_labels = [
        label.split("\n", 1)[1] if "\n" in label else label
        for label in experiment_labels
    ]

    rotate_xticks(ax, experiment_labels, rotation=45, ha="right")

    ax.yaxis.grid(True, linestyle=":", linewidth=0.7)
    ax.set_title("Execution Time Distribution (minutes) per Experiment")
    ax.set_xlabel(
        "Experiment (Configuration + Prompt Template; "
        "only Prompt Template shown as labels "
        "due to Configuration being communicated through hue)"
    )

    ax.set_ylabel("Execution Time (minutes)")


def rotate_xticks(ax, labels, rotation=45, ha="right"):
    """
    Rotate the x-axis tick labels for a given Axes object.
    """
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    return ax
