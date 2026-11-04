"""Small helpers shared across CLI entry points (e.g. ``~`` expansion on Path args)."""
from pathlib import Path

import pandas as pd

# The column resolve_embed_column writes when it prepends: "<prepend_col> <column>".
PREPENDED_COL = "_prepended_text"


def expand_path_args(args):
    """In-place: expand leading ``~`` on every Path-typed attr of an argparse Namespace."""
    for k, v in vars(args).items():
        if isinstance(v, Path):
            setattr(args, k, v.expanduser())
        elif isinstance(v, list) and v and all(isinstance(x, Path) for x in v):
            setattr(args, k, [x.expanduser() for x in v])
    return args


def resolve_embed_column(df: pd.DataFrame, column: str, prepend_col: str | None) -> str:
    """Return the column name to embed, prepending ``prepend_col`` (e.g. 'title') if present. A
    missing 'title' is skipped silently; any other missing prepend column raises.
    """
    if not prepend_col:
        return column
    if prepend_col not in df.columns:
        if prepend_col == "title":
            print(f"  --prepend-col 'title' not in columns; skipping (using {column!r} only)")
            return column
        raise ValueError(f"--prepend-col {prepend_col!r} not in columns: {list(df.columns)}")
    df[PREPENDED_COL] = (df[prepend_col].fillna("").str.strip() + " "
                         + df[column].fillna("").str.strip()).str.strip()
    return PREPENDED_COL


def embedded_text_col(df: pd.DataFrame, text_col: str = "text") -> str:
    """Read-side inverse of :func:`resolve_embed_column`: the column holding the exact string that was
    embedded, ``PREPENDED_COL`` when a prepend happened and ``text_col`` otherwise.
    """
    return PREPENDED_COL if PREPENDED_COL in df.columns else text_col
