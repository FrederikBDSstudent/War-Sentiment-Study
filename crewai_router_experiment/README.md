# CrewAI Router Experiment

This folder contains the small CrewAI routing experiment used to test whether a router can improve the saved LSTM benchmark predictions without changing the number of predictions.

Run:

```bash
python run_crewai_lstm_routing_experiment.py
```

The script uses the saved Market LSTM and Sentiment LSTM predictions, adds market/news context from the processed panels, and tests CrewAI-style routing agents. The main result keeps full coverage: `718 / 718` predictions.

| Model | Overall | Pre-war | War |
|---|---:|---:|---:|
| Market LSTM | `51.25%` on `718 / 718` | `61.45%` on `83 / 83` | `60.67%` on `89 / 89` |
| Sentiment LSTM | `55.43%` on `718 / 718` | `60.24%` on `83 / 83` | `69.66%` on `89 / 89` |
| Market-Sentiment Blend | `53.48%` on `718 / 718` | `63.86%` on `83 / 83` | `70.79%` on `89 / 89` |
| CrewAI Calibrated Router | `61.56%` on `718 / 718` | `65.06%` on `83 / 83` | `75.28%` on `89 / 89` |

Outputs are written to `artifacts/`.

The router shown above was selected from multiple candidates using the same overall evaluation period shown in the table. Its reported accuracy is therefore exploratory: evaluating the selected approach on a new untouched period is needed to assess whether the gain generalizes. The `deployment_safe` label in the output describes the candidate procedure; it does not correct for this selection.
