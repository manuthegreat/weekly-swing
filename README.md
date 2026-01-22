# Weekly Swing

This repo runs a daily batch pipeline that downloads Yahoo Finance daily bars, refreshes a rolling cache, runs the weekly swing backtest, and writes artifacts that a Streamlit dashboard can display.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the daily pipeline

```bash
python -m src.pipeline
```

To force a rerun (ignores per-day guard):

```bash
python -m src.pipeline full
```

Artifacts are written under `artifacts/<country>/` and rolling OHLC caches live in `cache_daily/`.

## Run the Streamlit app

```bash
streamlit run app.py
```

The app only reads artifacts and cached OHLC data. Run the pipeline at least once so the app has data to display.

## Git LFS for parquet outputs

Daily artifacts and cache files can grow quickly. This repo tracks parquet outputs with Git LFS to keep Git pushes reliable. If you are committing artifacts locally, install Git LFS and run:

```bash
git lfs install
```

## Daily GitHub Action

The workflow in `.github/workflows/daily_run.yml` runs the pipeline once per day and commits updated artifacts/cache data back to the repository.
