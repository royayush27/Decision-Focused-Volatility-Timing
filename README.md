# Decision-Focused Volatility Timing — Team 6

Submit the three numbered folders. They contain the project report, all code and input data, and the responses to the 21 questions. No presentation or development notes are required.

## Reproduce the results

Use Python 3.11 or later. From `2_Code_and_Data/DFVT_code_and_data`:

```sh
python -m pip install -r requirements.txt
python report_extras.py --paper
python qa_checks.py
```

The first script executes all analysis notebook code and recreates the report tables and figures. The second executes the analysis and prints the additional Q&A checks. To save its output, run `python qa_checks.py > qa_checks_output.txt`. These are substantial bootstrap runs; allow several minutes or longer depending on the computer.

`vol_timing_final_revised.ipynb` is the analysis. `lstm_training_revised.ipynb` recreates the six supplied forecast exports; its full six-factor, five-seed run takes many hours. Keep `SMOKE_TEST = False` for the reported experiment. Copy the six completed factor CSVs from its run-id directory to `data/LSTM-Forecasts-revised/` before rerunning analysis. `nb_loader.py` is required by both scripts.

The four base CSV files and six forecast CSVs are required. `data/Data_Collection.ipynb` and `data/raw/` preserve the French-data preparation inputs. `cache/` avoids refitting classical GARCH; it is optional for correctness. Generated figures, result tables and logs are recreated by the commands above and are omitted from main. The report's LaTeX figure copies are retained to rebuild the PDFs.

## Replication caveats

This is a pseudo-out-of-sample backtest: decisions use earlier dates, but Ken French returns are revised historical data, not archived real-time vintages. VIX before 2003 was back-calculated. The start date, cap and comparison rules were explored on this dataset; the reported multiple-testing adjustments do not cover that entire exploration. Results therefore do not constitute independent prospective validation.

The licensed course file `IFE_term_stock_data.csv` is not included. Place it in `2_Code_and_Data/DFVT_code_and_data/data/` to reproduce the CRSP benchmark and the corresponding Question 9 sensitivity checks. Without it, those checks are skipped; the main market and factor tests still run. Exact replication of every reported benchmark requires access to that course file.

Full neural-network retraining can differ slightly across PyTorch versions and hardware. The supplied forecasts reproduce the reported analysis without retraining. `requirements.txt` gives compatible minimum versions, not a frozen original training environment.

## Rebuild the documents

From `1_Project_Report/latex_source`, compile `main.tex` and `responses.tex` with Tectonic, or run pdfLaTeX twice for each. Copy `main.pdf` to `../DFVT_project_report.pdf` and `responses.pdf` to `../../3_Responses_to_Questions/DFVT_responses_to_questions.pdf`.

## Branches

`main` contains the lean submission. `DFVT_Older-Version` preserves the previous GitHub main (commit `533a3c2`). `main` contains only this submission and its replication sources. The enclosing local project's index and working files were left untouched. Repository: https://github.com/royayush27/Decision-Focused-Volatility-Timing.

## Verification

The analysis, report-figure script and all available Q&A checks were rerun on Python 3.12 (NumPy 2.3.1, pandas 2.2.3, SciPy 1.15.3, Matplotlib 3.10.8). The Q&A output matches the saved numerical results; the core performance table reproduces. Training accounting and temporal-split checks passed with PyTorch 2.11.0. The complete multi-hour neural training and licensed CRSP checks were not rerun. Small numerical differences can arise when a GARCH cache is refitted.
