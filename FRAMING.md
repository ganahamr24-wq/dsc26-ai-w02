## Decision
Dina and her 5 analysts need to review flagged transactions efficiently without being overwhelmed by false positives. Setting an optimal threshold directly controls the daily queue size and analyst workload.

## Metric
We optimize for expected financial cost and precision at threshold, ensuring high precision so analysts process genuinely high-risk cases while minimizing missed fraud (false negatives).

## Baseline
A simple baseline model or standard rule-based filter (e.g., flagging transactions above a specific amount or score quantile) serves as the benchmark before deploying complex ML scoring.

## Classical ML or LLM
Classical ML models like Random Forest or XGBoost are preferred for tabluar fraud detection due to latency constraints and strong tabular performance, while LLMs are better suited for text/explanation tasks.