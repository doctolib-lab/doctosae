"""Parse a training sweep .sh grid and load its completed run results.

Runs are matched by reading each run folder's own ``summary.json["config"]``, never by parsing
the run folder name. Non-swept flags (explicit or defaulted) are matched too, so two sweeps sharing
a runs_dir but differing in a fixed flag never mix each other's runs. Flags typed ``Path`` (data
locations) are left out of that match: a run stores them expanded and absolute, so they need not
equal the text of the sweep script.
"""
from __future__ import annotations

import ast
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "training" / "train_sae.py"


def _flag_to_config_key(train_script: Path = TRAIN_SCRIPT) -> dict[str, dict]:
    """Map each ``--flag`` of train_sae.py's argparse to {"key": dest, "default": ..., "store_true": bool,
    "is_path": bool}, where "is_path" marks a ``type=Path`` flag (a data location, not a hyperparameter)."""
    text = Path(train_script).read_text()
    mapping = {}
    for chunk in text.split("p.add_argument(")[1:]:
        fm = re.match(r'\s*f?"--([\w-]+)"', chunk)
        if not fm:
            continue
        flag = fm.group(1)
        stmt_end = chunk.find(")\n")
        stmt = chunk[:stmt_end] if stmt_end != -1 else chunk
        dm = re.search(r'dest="([\w_]+)"', stmt)
        key = dm.group(1) if dm else flag.replace("-", "_")
        store_true = 'action="store_true"' in stmt
        if store_true:
            default = False
        else:
            dm2 = re.search(r'default=([^,)]+)', stmt)
            try:
                default = ast.literal_eval(dm2.group(1).strip()) if dm2 else None
            except (ValueError, SyntaxError):
                default = dm2.group(1).strip() if dm2 else None
        mapping[flag] = {"key": key, "default": default, "store_true": store_true,
                         "is_path": "type=Path" in stmt}
    return mapping


def _swept_config_keys(text: str, flag_to_key: dict[str, dict]) -> dict[str, str]:
    """Map each swept bash loop var (e.g. LAM) to its config.json key (e.g. lambda_), from the
    script's ``for VAR in ...;`` loops cross-referenced with the ``--flag "$VAR"`` it feeds.
    """
    loop_vars = re.findall(r'for ([A-Za-z_][A-Za-z_0-9]*) in ', text)
    var_to_key = {}
    for var in loop_vars:
        m = re.search(rf'--([\w-]+)\s+"\$\{{?{var}\}}?"', text)
        if m and m.group(1) in flag_to_key:
            var_to_key[var] = flag_to_key[m.group(1)]["key"]
    return var_to_key


def _fixed_config(text: str, resolved_text: str, flag_to_key: dict[str, dict], swept_keys: set[str]) -> dict:
    """(expected, true_default) for every train_sae.py config key this sweep does not vary. A run
    missing the key is compared against ``true_default``. Path flags are skipped: they locate the
    data, and a run stores them expanded, so the script text cannot be compared with them.
    """
    fixed = {}
    for flag, info in flag_to_key.items():
        key = info["key"]
        if key in swept_keys or info.get("is_path"):
            continue
        if info["store_true"]:
            expected = bool(re.search(rf'--{re.escape(flag)}\b', text)) or info["default"]
        else:
            m = re.search(rf'--{re.escape(flag)}\s+"?([^\s"]+)"?', resolved_text)
            expected = parse_hparam_value(m.group(1)) if m else info["default"]
        fixed[key] = (expected, info["default"])
    return fixed


def _swept_values(text: str, var: str) -> list[str]:
    """Raw string values from a ``for VAR in v1 v2 ...; do`` loop."""
    m = re.search(rf'for {re.escape(var)} in ([^;]+);', text)
    return m.group(1).split() if m else []


def _parse_data_paths(text: str) -> dict:
    """Resolve simple bash variables and extract data file paths from a sweep script."""
    vars: dict[str, str] = {}
    for m in re.finditer(r'^([A-Za-z_][A-Za-z_0-9]*)=([^\n]+)', text, re.MULTILINE):
        name, val = m.group(1), m.group(2).strip().strip('"\'')
        val = re.sub(r'\$\{?([A-Za-z_][A-Za-z_0-9]*)\}?',
                     lambda mv: vars.get(mv.group(1), mv.group(0)), val)
        val = val.replace('~', str(Path.home()))
        vars[name] = val

    def _resolve(s: str) -> str:
        return re.sub(r'\$\{?([A-Za-z_][A-Za-z_0-9]*)\}?',
                      lambda mv: vars.get(mv.group(1), mv.group(0)), s).replace('~', str(Path.home()))

    def _flag_path(flag: str) -> Path | None:
        m = re.search(rf'--{re.escape(flag)}\s+"([^"]+)"', text)
        if not m:
            m = re.search(rf'--{re.escape(flag)}\s+(\S+)', text)
        return Path(_resolve(m.group(1))) if m else None

    def _flag_val(flag: str) -> str | None:
        m = re.search(rf'--{re.escape(flag)}\s+(\S+)', text)
        return m.group(1).strip('"\'') if m else None

    return {
        "test_queries":  _flag_path("test-queries"),
        "test_qrels":    _flag_path("test-qrels"),
        "train_corpus":  _flag_path("train-corpus"),
        "val_corpus":    _flag_path("val-corpus"),
        "test_corpus":   _flag_path("test-corpus"),
        "val_score_train_size": _flag_val("val-score-train-size"),
        "corpus_id_col": _flag_val("corpus-id-col"),
        "query_id_col":  _flag_val("query-id-col"),
        "score_col":     _flag_val("score-col"),
    }


def parse_sweep(sh_path: Path, train_script: Path = TRAIN_SCRIPT) -> dict:
    """Parse a sweep .sh file into {runs_dir, swept_keys, swept_values, test_queries, test_qrels,
    train_corpus, corpus_id_col, query_id_col, score_col}.
    """
    text = Path(sh_path).read_text()

    flag_to_key = _flag_to_config_key(train_script)
    var_to_key = _swept_config_keys(text, flag_to_key)
    swept_values = {key: _swept_values(text, var) for var, key in var_to_key.items()}

    # Resolve simple shell variable assignments (e.g. REPO=/path/prefix).
    resolved_text = text
    for var, value in re.findall(r'^(\w+)=(\S+)$', text, re.MULTILINE):
        resolved_text = resolved_text.replace(f"${{{var}}}", value).replace(f"${var}", value)

    # Extract runs dir from --experiment-folder "/path/prefix_${TAG}"
    m = re.search(r'--experiment-folder\s+"([^"$]+)\$\{TAG\}"', resolved_text)
    if not m:
        raise ValueError(f"Cannot find --experiment-folder in {sh_path}")
    runs_dir = Path(m.group(1)).parent

    swept_keys = set(var_to_key.values())
    result = {
        "runs_dir":      runs_dir,
        "swept_keys":    swept_keys,
        "swept_values":  swept_values,
        "fixed_config":  _fixed_config(text, resolved_text, flag_to_key, swept_keys),
    }
    result.update(_parse_data_paths(text))
    return result


def load_runs(runs_dir: Path, swept_keys: set[str], fixed_config: dict | None = None,
             swept_values: dict | None = None) -> dict:
    """Load summary.json for every completed run folder in runs_dir, as {signature: {seed: summary}}.

    The signature is a sorted tuple of (key, value) for every swept hyperparameter except seed.
    Values come from each run's own summary.json["config"]; ``fixed_config`` and ``swept_values``
    must match too. Each summary is stamped with "_run_dir".
    """
    other_keys = sorted(swept_keys - {"seed"})
    fixed_config = fixed_config or {}
    allowed = {key: {parse_hparam_value(v) for v in vals}
              for key, vals in (swept_values or {}).items() if vals}
    data: dict = defaultdict(dict)
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        sf = d / "summary.json"
        if not sf.exists():
            continue
        summary = json.loads(sf.read_text())
        cfg = summary.get("config", {})
        if "seed" not in cfg:
            continue
        if any(cfg.get(key, true_default) != expected
               for key, (expected, true_default) in fixed_config.items()):
            continue
        if any(key in cfg and cfg[key] not in vals for key, vals in allowed.items()):
            continue
        summary["_run_dir"] = str(d)
        # lambda_=0 zeroes the contrastive loss, so temperature is dropped from the signature.
        degenerate = {"temperature"} if cfg.get("lambda_") == 0 else set()
        signature = tuple((key, cfg[key]) for key in other_keys
                          if key in cfg and key not in degenerate)
        data[signature][cfg["seed"]] = summary
    return data


def parse_hparam_value(raw: str):
    """Parse a CLI string into the type it'd have in a run's config (float when possible)."""
    try:
        return float(raw)
    except ValueError:
        return raw


def filter_by_hparam(data: dict, key: str, value) -> dict:
    """Restrict {signature: {seed: summary}} to configs where signature[key] == value.
    A signature without ``key`` is kept: the hyperparameter doesn't apply to it."""
    filtered = {sig: seeds for sig, seeds in data.items()
                if dict(sig).get(key, value) == value}
    if not filtered:
        raise ValueError(f"No runs found with {key}={value}")
    return filtered


def values_of(data: dict, key: str) -> list:
    """Sorted distinct values ``key`` takes across the configs in ``data``."""
    return sorted({dict(sig)[key] for sig in data if key in dict(sig)})


def warn_if_not_swept(grid: dict, key: str, value) -> None:
    """Warn if `value` isn't among the values the sweep .sh actually looped over for `key`."""
    swept = grid.get("swept_values", {}).get(key)
    if swept is None:
        return
    parsed = [parse_hparam_value(v) for v in swept]
    if value not in parsed:
        warnings.warn(f"--filter {key}={value!r} not among values swept in the .sh script: {parsed}")


def val_metric_from(metric: str) -> str:
    """Derive val metric name from test metric: 'test_auc' -> 'best_val_auc'."""
    return "best_val_" + metric.split("test_", 1)[1]


def mean_sem(values) -> tuple[float, float, int]:
    """(mean, SEM, n), with SEM = sample SD (ddof=1) / sqrt(n). Raises ValueError at n < 2.
    """
    v = np.asarray(list(values), dtype=float)
    if v.size < 2:
        raise ValueError(f"need >= 2 samples to compute a SEM, got {v.size}")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)


def format_mean_sem(mean: float, sem: float, n: int) -> str:
    """e.g. (149.24467, 2.0806, 10) -> '149 ± 2.1', by Clymo's significant-digit rules: the mean's
    last significant digit sits in the SEM's first significant digit's decade, and the SEM's digit
    count follows n.
    """
    if not np.isfinite(sem) or sem <= 0:
        return f"{mean:g} ± {sem:g}"
    sem_digits = next((d for lim, d in ((6, 1), (100, 2), (10_000, 3), (10 ** 6, 4)) if n <= lim), 5)
    decade = int(np.floor(np.log10(sem)))  # decade of the SEM's first significant digit
    return f"{mean:.{max(0, -decade)}f} ± {sem:.{max(0, sem_digits - 1 - decade)}f}"


def ci_half_width(sem: float, n: int) -> float:
    """The 95% CI half-width: t(0.975, n-1) * SEM, the quantile from n, not a fixed 1.96."""
    return float(t_dist.ppf(0.975, n - 1)) * sem


def format_mean_ci(mean: float, sem: float, n: int, decimals: int = 3) -> str:
    """'mean' then the 95% CI half-width in scriptsize, for LaTeX, both at `decimals` digits."""
    return (f"{mean:.{decimals}f}"
            f"{{\\scriptsize$\\,\\pm\\,${ci_half_width(sem, n):.{decimals}f}}}")


def ci_indistinguishable(mean_a: float, sem_a: float, n_a: int,
                         mean_b: float, sem_b: float, n_b: int) -> bool:
    """Whether two means' 95% CIs overlap -- the intervals the tables print."""
    ci_a, ci_b = ci_half_width(sem_a, n_a), ci_half_width(sem_b, n_b)
    return mean_a - ci_a <= mean_b + ci_b and mean_b - ci_b <= mean_a + ci_a


def config_stats(data: dict, val_metric: str) -> dict:
    """{signature: (mean, SEM, n)} over each config's seeds. Skips configs lacking the metric, and
    raises ValueError naming a config left with a single seed.
    """
    out = {}
    for sig, seeds in data.items():
        vals = [s[val_metric] for s in seeds.values() if val_metric in s]
        if not vals:
            continue
        try:
            out[sig] = mean_sem(vals)
        except ValueError:
            raise ValueError(f"config {dict(sig)} has only {len(vals)} seed with {val_metric} — "
                             f"rerun once every seed of the sweep has completed") from None
    return out


def best_config(data: dict, val_metric: str, fixed: dict | None = None) -> tuple:
    """Signature of the best config among those matching every key=value in ``fixed``: the smallest
    hidden among the configs whose val-metric 95% CI overlaps that of the highest-mean one.
    """
    for key, value in (fixed or {}).items():
        data = filter_by_hparam(data, key, value)
    stats = config_stats(data, val_metric)
    if not stats:
        raise ValueError(f"No config has {val_metric}")
    mean_winner = max(stats, key=lambda sig: stats[sig][0])
    tied = [sig for sig in stats if ci_indistinguishable(*stats[mean_winner], *stats[sig])]
    if any("hidden" not in dict(sig) for sig in tied):
        raise ValueError("config selection needs 'hidden' in every signature — it is not swept here")
    return min(tied, key=lambda sig: (dict(sig)["hidden"], -stats[sig][0]))


def best_per_value(data: dict, val_metric: str, key: str, *, fixed: dict | None = None) -> dict:
    """{value: signature} — best_config once per distinct value ``key`` takes in ``data``. ``fixed``
    pins further hyperparameters for every value; values with no run carrying ``val_metric`` are
    dropped.
    """
    out = {}
    for value in values_of(data, key):
        subset = filter_by_hparam(data, key, value)
        for fixed_key, fixed_value in (fixed or {}).items():
            subset = {sig: seeds for sig, seeds in subset.items()
                      if dict(sig).get(fixed_key) == fixed_value}
        if any(val_metric in s for seeds in subset.values() for s in seeds.values()):
            out[value] = best_config(subset, val_metric)
    return out


def load_sweep_runs(sweep_sh: Path, runs_dir: Path | None = None) -> tuple[dict, dict]:
    """(grid, data) for a sweep .sh — parse_sweep + load_runs, with runs_dir optionally overridden
    when finished runs have been moved off the script's own --experiment-folder path.
    """
    grid = parse_sweep(sweep_sh)
    if runs_dir is not None:
        grid["runs_dir"] = runs_dir
    data = load_runs(grid["runs_dir"], grid["swept_keys"], grid["fixed_config"], grid["swept_values"])
    if not data:
        raise ValueError(f"No completed runs found in {grid['runs_dir']}")
    return grid, data


def _report_row(sig: tuple, seeds: dict, val_metric: str, metric: str) -> str:
    """One tab-separated line describing a selected config: hparams, val, test, seeds, run dirs."""
    hparams = ",".join(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}"
                       for key, value in sig)
    cells = [hparams]
    for name in (val_metric, metric):
        vals = [s[name] for s in seeds.values() if name in s]
        cells.append(f"{name}={format_mean_sem(*mean_sem(vals))}" if len(vals) > 1
                     else f"{name}=" + (f"{vals[0]:g}" if vals else "n/a"))
    cells.append(f"seeds={','.join(str(s) for s in sorted(seeds))}")
    cells.append(",".join(seeds[s]["_run_dir"] for s in sorted(seeds)))
    return "\t".join(cells)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Print the best config of a sweep, selected by validation metric. One "
                    "tab-separated line per selected config: hparams, val, test, seeds, run dirs.")
    p.add_argument("--sweep",    required=True, type=Path, help="Sweep .sh script path")
    p.add_argument("--metric",   required=True, help="Test metric (e.g. test_ndcg, test_auc); the "
                                                    "val metric to select on is derived from it")
    p.add_argument("--runs-dir", type=Path, default=None, help="Override runs_dir from the sweep script")
    p.add_argument("--per", default=None, metavar="KEY",
                   help="Report the best config for each value of KEY (e.g. --per k). Omit for a "
                        "single overall best")
    p.add_argument("--filter", nargs=2, metavar=("HPARAM", "VALUE"), action="append", default=None,
                   help="Pin a hyperparameter, e.g. --filter k 128. Repeatable")
    args = p.parse_args()

    grid, data = load_sweep_runs(args.sweep, args.runs_dir)
    fixed = {key: parse_hparam_value(value) for key, value in (args.filter or [])}
    for key, value in fixed.items():
        warn_if_not_swept(grid, key, value)
    val_metric = val_metric_from(args.metric)

    if args.per:
        if args.per in fixed:
            p.error(f"--per {args.per} conflicts with --filter {args.per}")
        selected = best_per_value(data, val_metric, args.per, fixed=fixed).items()
    else:
        selected = [(None, best_config(data, val_metric, fixed))]
    for value, sig in selected:
        prefix = f"{args.per}={value:g}\t" if isinstance(value, float) else \
                 (f"{args.per}={value}\t" if value is not None else "")
        print(prefix + _report_row(sig, data[sig], val_metric, args.metric))
