# 8 Summary of Findings

This section summarizes key findings from the sentiment analysis experiments, focusing on model performance,
prompt development, and insights gained.

## 8.1 Model Performance

- **Accuracy Variability**: Accuracy fluctuated significantly across models, prompts, and temperatures, indicating
  that a larger dataset is needed to reduce variance and provide reliable insights.

- **PHI Model Speed**: The PHI model is consistently slower than QWEN, suggesting smaller models like QWEN
  are worth exploring for better speed without sacrificing accuracy.

## 8.2 Prompt Development

- **Iterative Refinement**: Iterating on prompts by analyzing mistakes on single reviews quickly highlights specific issues
  in prompt design, allowing for targeted improvements.

- **Need for Clear Sentiment Definition**: Sentiment should be clearly defined to prevent confusion between execution and
  subject sentiment, particularly when models review episodes versus entire series.

- **Dataset Ambiguity**: The dataset lacks clarity on whether a review pertains to a single season or an entire series,
  leading to potential inconsistencies in sentiment labeling.
