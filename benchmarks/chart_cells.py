"""Shared stale-aware loading for benchmark presentation paths.

The per-cell store is append/resume oriented, so it can legitimately contain a
mixture of measurements from several revisions.  Presentation code must never
interpret that mixture directly: ``stale_cells.py`` is the authoritative list
of semantic/timing changes that invalidate an older record.

Current renderers either filter stale records (for a partially refreshed fair
store) or reject the input (for an exact-denominator scaling store).  Historical
snapshot renderers are deliberately not routed through this module; they must
identify their pinned source as historical at their own entry point.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import runner_common as rc
import stale_cells


class CellStoreError(RuntimeError):
    """A current chart source cannot be interpreted safely."""


class StaleCellError(CellStoreError):
    """A reject-on-stale presentation path received invalidated records."""


@dataclass(frozen=True)
class CellEntry:
    path: Path
    record: dict


@dataclass(frozen=True)
class CellLoadReport:
    source: str
    selected: int
    current: int
    stale: int
    non_cells: int


@cache
def _sha_contains(sha: str, required_commit: str) -> bool:
    """Whether a recorded revision contains one stale-rule fix.

    Unknown/missing revisions fail closed.  That matches ``stale_cells.py``:
    without ancestry evidence, a scoped old record is not publishable as current.
    """
    if not sha:
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_commit, sha],
        cwd=rc.ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


@lru_cache(maxsize=1)
def _registered_kinds() -> dict[str, str]:
    return stale_cells.kind_of_matrix()


def stale_reasons(record: dict) -> tuple[str, ...]:
    """Return the same invalidation reasons as ``stale_cells.py`` for one cell."""
    reasons = []
    if stale_cells.timeout_cap_is_stale(record):
        reasons.append(stale_cells.TIMEOUT_CAP_REASON)

    cell = record.get("cell") or {}
    solver = cell.get("solver", "")
    matrix_id = cell.get("matrix_id", "")
    device = cell.get("device", "")
    kind = (record.get("matrix_meta") or {}).get("kind")
    if kind is None:
        kind = _registered_kinds().get(matrix_id)
    sha = (record.get("provenance") or {}).get("git_sha", "")
    for commit, reason in stale_cells.matching_rules(
            solver, matrix_id, kind, device):
        if not _sha_contains(sha, commit):
            reasons.append(f"{commit}: {reason}")
    return tuple(reasons)


def _paths(root: str | Path, pattern: str) -> Iterable[Path]:
    root = Path(root)
    if root.is_file():
        return (root,)
    return sorted(root.glob(pattern))


def load_current_entries(
    root: str | Path,
    *,
    pattern: str = "**/*.json",
    include: Callable[[dict], bool] | None = None,
    stale_policy: str = "filter",
    source: str = "chart cell store",
    announce: bool = True,
) -> tuple[list[CellEntry], CellLoadReport]:
    """Load provenance-bearing cells after applying every current stale rule.

    ``stale_policy='filter'`` supports resume-safe fair stores: current cells are
    rendered and invalidated ones become honest gaps. ``'reject'`` is for paths
    with an exact expected denominator, where silently shrinking the input would
    be misleading.
    """
    if stale_policy not in {"filter", "reject"}:
        raise ValueError(f"unknown stale policy {stale_policy!r}")

    current: list[CellEntry] = []
    stale: list[tuple[Path, tuple[str, ...]]] = []
    selected = 0
    non_cells = 0
    for path in _paths(root, pattern):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CellStoreError(f"invalid JSON cell {path}: {error}") from error
        if not isinstance(record, dict) or "cell" not in record or "status" not in record:
            non_cells += 1
            continue
        if include is not None and not include(record):
            continue
        selected += 1
        reasons = stale_reasons(record)
        if reasons:
            stale.append((path, reasons))
        else:
            current.append(CellEntry(path, record))

    report = CellLoadReport(source, selected, len(current), len(stale), non_cells)
    summary = (f"{source}: checked {selected}/{selected} selected cells; "
               f"current={len(current)}, stale={len(stale)}")
    if announce:
        print(summary, file=sys.stderr)
    if stale and stale_policy == "reject":
        examples = "; ".join(
            f"{path}: {reasons[0]}" for path, reasons in stale[:3]
        )
        extra = f"; plus {len(stale) - 3} more" if len(stale) > 3 else ""
        raise StaleCellError(f"{summary}; refusing current render/export: {examples}{extra}")
    return current, report


def load_current_records(*args, **kwargs) -> tuple[list[dict], CellLoadReport]:
    entries, report = load_current_entries(*args, **kwargs)
    return [entry.record for entry in entries], report
