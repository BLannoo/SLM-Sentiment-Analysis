# 9 Future Work and Next Steps

To further enhance the sentiment analysis framework, several key next steps are suggested.
These focus on refining prompt strategies, improving dataset quality, comparing with LLMs, and ensuring robustness through testing.

## 9.1 Validate and Refine Prompt Templates

- Focus on prompts that performed poorly, such as the "debate" style prompts.
- Ensure each prompt is the best possible representation of its intended strategy.
- Use insights from single review analyses to guide refinements.

## 9.2 Increase Dataset Size for Stability

- Expand the dataset size from 100 to 1,000 reviews per experiment to reduce accuracy variance.
- Move the evaluation to a new subset of the data that was never looked at during iterative design.

## 9.3 Compare with Large Language Models (LLMs)

- Introduce LLMs like GPT-3/4 into the experiments to compare against the small language models (SLMs).


## 9.4 Improve Testing for Robustness

- Implement unit and integration tests to ensure all analysis scripts and pipelines are bug-free.
