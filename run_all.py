"""Run scripts 1-7 and 9, then 8 (multiple testing collects all of them) for one spec, then make_figures.py (main spec). Each script writes its own log, tables and net returns to results/<spec>/.
The first run fits every GARCH model (roughly 15-20 minutes); later runs reuse results/cache/ and take about 2 minutes.
Usage:  python run_all.py                  (main spec, the one in SPEC.md)
        python run_all.py --spec old       (our earlier settings, robustness)
        python run_all.py --spec uncapped  (no leverage cap, robustness)"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = sys.argv[sys.argv.index("--spec") + 1] if "--spec" in sys.argv else "main"
SCRIPTS = ["1_us_market", "2_us_each_series", "3_us_factor_weighting", "4_us_factor_combos", "5_us_main_result",
           "6_intl_jkp", "7_intl_french", "9_dfl_robin", "8_multiple_testing"]   # 8 last: it collects the results of all others
for s in SCRIPTS:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / f"{s}.py"), "--spec", spec], capture_output=True, text=True, cwd=ROOT)
    print(f"{s:24s} {'ok' if r.returncode == 0 else 'FAILED'}  {time.time() - t0:5.0f}s", flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:]); sys.exit(1)
if spec == "main":                                                        # figures for the README and slides
    for extra in ([], ["--slides"]):
        r = subprocess.run([sys.executable, str(ROOT / "make_figures.py"), *extra], capture_output=True, text=True, cwd=ROOT)
        print(f"{'make_figures ' + ' '.join(extra):24s} {'ok' if r.returncode == 0 else 'FAILED'}", flush=True)
print(f"done: results in {ROOT / 'results' / spec}")
