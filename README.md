# README

This project aims to explore sentiment analysis on movie reviews using various prompt engineering strategies.
The analysis involves running experiments with different models, temperatures, prompt templates, and device types
to evaluate performance based on accuracy and execution time.

## Setup

### 1. Local Environment Setup

Instructions to set up the project environment locally, including installing dependencies, configuring virtual environments,
and preparing the necessary directories.

For more details, check [001_setup_local_environment.md](docs/001_setup_local_environment.md).

### 2. Google Colab Setup

Guidelines to run experiments on Google Colab, focusing on setting up secrets, managing dependencies, and
configuring the environment for GPU usage to speed up execution.

For more details, check [002_setup_google_collab.md](docs/002_setup_google_collab.md).

## Project Workflow

### 3. Data Preparation

Details about the dataset used, its preparation steps, and caching strategies. This section also covers how the
data is processed and balanced to ensure a fair experiment setup.

For more details, check [003_data_preparation.md](docs/003_data_preparation.md).

### 4. Running Experiments

Explains how to run batch sentiment analysis experiments using different models, temperatures, and prompt templates.
It includes details on the configuration options and usage examples.

For more details, check [004_running_experiments.md](docs/004_running_experiments.md).

### 5. First Analysis of Experiment Results

Describes the initial steps to analyze experiment results using provided scripts. This section covers generating summary
statistics and deeper insights for individual review assessments.

For more details, check [005_first_analysis_of_experiment_results.md](docs/005_first_analysis_of_experiment_results.md).

### 6. Prompt Engineering and Design

Explores the iterative process of designing and refining prompts to improve model performance. It also discusses strategies
used for creating effective prompts and the logic behind them.

For more details, check [006_prompt_engineering_and_design.md](docs/006_prompt_engineering_and_design.md).

## Conclusion

### 7. Evaluation Metrics and Analysis

Focuses on evaluating the models and prompts based on key metrics such as accuracy and execution time.
Includes visual representations of the results and comparisons across different configurations.

For more details, check [007_evaluation_metrics_and_analysis.md](docs/007_evaluation_metrics_and_analysis.md).

### 8. Summary of Findings

Summarizes the key findings from the experiments, focusing on model performance, prompt development,
and insights on defining sentiment and dataset ambiguity. It highlights the need for a larger dataset,
exploration of smaller models, and clearer sentiment definitions to enhance future analysis.

For more details, check [008_summary_of_findings.md](docs/008_summary_of_findings.md).

### 9. Future Work and Next Steps

Identifies potential areas for future exploration, such as improving prompt templates, validating strategies,
and comparing results against larger language models.

For more details, check [008_future_work_and_next_steps.md](docs/009_future_work_and_next_steps).
