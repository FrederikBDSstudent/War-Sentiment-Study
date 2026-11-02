# Study workflow

The study is run through three notebooks:

1. `00_data_collection.ipynb` builds the filing, event, and market panels using `src/oil_projects`.
2. `01_training_lstm_pair.ipynb` trains the matched LSTM models and writes predictions, metrics, and reference artifacts.
3. `02_analysis_war_sentiment.ipynb` reads the saved panels and artifacts to produce the analysis.

`data/processed/` contains inputs and caches used by those workflows. `artifacts/` contains the saved benchmark outputs and reference models. `assets/` contains the README figures. The optional GDELT collection path is retained because the collection notebook explicitly offers it; the selected benchmark does not enable its enriched feature set.

Unused legacy model runners and the duplicate training module were removed. The standalone `crewai_router_experiment/` is retained because its saved comparison reports higher accuracy than the selected LSTM blend. Gemini, Groq, and other local experiments were moved outside this repository into the local `war_sentiment_study_experiment_archive_20261102` folder. The CrewAI router candidate is selected using overall results across the same evaluated period, so its reported gain is exploratory and needs a separate untouched evaluation period.

Before the final repository cleanup, all three notebooks were executed on isolated copies before and after the code cleanup. The runs produced 35 identical output files and 13 identical analysis figures. Historical saved results were preserved. The fresh training run did not reproduce the historical benchmark in either version; its cause was not isolated. The saved news panel's sentiment fields are zero, so the benchmark's interpretation of sentiment requires further investigation.
