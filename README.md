# KETEP Power Forecasting

Data center power demand forecasting experiments.

## Files

- `dc_forecast_experiments.py`: core experiments for E0, E1, E2, E3, and E8.
- `main.py`: model training/evaluation entry point.
- `main_individual_datacenter.py`: per-data-center forecasting entry point.
- `dc_customers_2025_utf8_bom.csv`: customer metadata used by the experiments.

## Local Data

Large local datasets and generated outputs are intentionally excluded from git.

- `data.csv`
- `exp_out/`
- `forecast_outputs_gpu/`
- `forecast_outputs_individual/`
- `lightning_logs/`

Place the hourly load data as `data.csv` before running `dc_forecast_experiments.py`.

## Run

```bash
python dc_forecast_experiments.py
```
