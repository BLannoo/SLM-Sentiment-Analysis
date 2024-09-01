# 6 Prompt Engineering and Design

This section describes the process of designing and iterating on prompts for sentiment analysis
to maximize model performance. The goal is to create prompts that consistently yield accurate
sentiment classifications while being interpretable by the sentiment parsing logic.

## 6.1 Initial Prompt Framework

We began with a basic prompt framework designed to generate outputs that could be easily parsed
by a sentiment extraction method. The primary sentiment parsing relies on a regex pattern
to identify the sentiment within the output:

```python
re.compile(
    rf"(?i)Execution\s*sentiment:\s*-?\s*({'|'.join(allowed_sentiments)})\b",
    re.DOTALL,
)
```

For additional fallback mechanisms, refer to the code in
[sentiment_parser.py](../src/model/sentiment_parser.py).

A basic example prompt that adheres to this framework is
[001-direct-instruction.txt](../prompts/strategies/001-direct-instruction.txt):

```plaintext
**Review:**
{review}

**Instructions:**

Classify the execution sentiment of the movie review, <<provide the strategy specific instructions>>

**Output Format:**

Reasoning:
<summary of the thinking steps and reflections done as described by the instructions>

Execution sentiment: [One of 'Very positive', 'Positive', 'Mixed positive', 'Mixed negative', 'Negative', 'Very negative']
````

### 6.1.1 Iterative Prompt Improvement

By running batch experiments using prompts like **001-direct-instruction**,
we identified instances of poor classification and iterated to create improved versions,
such as [015-weighted-sentiment-filtering.txt](../prompts/strategies/015-weighted-sentiment-filtering.txt).

## 6.2 Generating New Prompts from Strategies

We leveraged ChatGPT-4o to survey existing prompt strategies documented online and supplemented
this with additional strategies not initially covered, such as debate-style prompts.
For each identified strategy, we generated a corresponding prompt template and ran initial
batch experiments to assess their effectiveness.

### 6.2.1 Meta-Prompt for Refinement

To refine the weakest-performing prompts, we created a **meta-prompt** in
[meta-prompt.txt](../prompts/strategies/meta-prompt.txt) that guided the generation
of improved versions while retaining the core strategy.

### 6.2.2 First Iteration and Future Refinements

We completed one round of prompt improvements using the meta-prompt but had limited time
for further iterations. Future work could focus on refining the weaker strategies identified
in the [First Analysis of Experiment Results](../docs/005_first_analysis_of_experiment_results.md).
For example, the debate prompts might have potential but are poorly represented by the current prompt,
which allows the experts to talk about generic movie review sentiments rather than the ones in the current review.

**Note**: Further changes to prompts would require significant computational resources,
which are beyond the current scope.

## 6.3 Conclusion

Prompt engineering is an iterative process. Starting with a well-defined framework and refining
based on model performance allows for continuous improvement. Future iterations should focus on
the worst-performing prompts, applying targeted refinements to enhance accuracy and reliability.
