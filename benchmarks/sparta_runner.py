"""Optional SPARTA subprocess on the Metal axis, using shared exported A/b."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import runner_common as rc

PYTHON = os.environ.get("APXCHOL_SPARTA_PYTHON", "")
WORKER = Path(__file__).with_name("sparta") / "worker.py"


def available():
    if sys.platform != "darwin":
        return False, "SPARTA Metal requires macOS"
    if not PYTHON or not os.path.isfile(PYTHON) or not os.access(PYTHON, os.X_OK):
        return False, "set APXCHOL_SPARTA_PYTHON to the pinned SPARTA environment's Python"
    return True, ""


def summarize(records, returncode, repeats, warmup, tol, timed_out=False):
    receipts = [f"BENCH_REPEAT phase={'retained' if r.get('retained') else 'warmup'} "
                f"index={r.get('repetition')} total_s={r['total_s']}"
                for r in records if isinstance(r.get("total_s"), (float, int))]
    last = records[-1] if records else {}
    measured = [r for r in records if r.get("status") in ("complete", "not_converged")]
    for index, r in enumerate(measured):
        if (r.get("execution_route") != "metal" or r.get("stop_contract") != "original-v1"
                or r.get("provenance", {}).get("sparta_contract") != rc.SPARTA_CONTRACT
                or r.get("provenance", {}).get("sparta_revision") != rc.SPARTA_REVISION
                or type(r.get("repetition")) is not int or r["repetition"] != index
                or type(r.get("retained")) is not bool or r["retained"] != (index >= warmup)
                or index >= warmup + repeats):
            return "failed", {}, {"na_reason": "invalid SPARTA route/source/repetition record"}
    if timed_out:
        provenance = measured[-1]["provenance"] if measured else {}
        return "timeout", {"repeat_receipts": receipts, "partial_repetitions": records,
                           "sparta_provenance": provenance}, {
            "na_reason": "SPARTA process deadline exceeded", "timeout_scope": "logical_cell"}
    if last.get("status") == "n/a":
        return "n/a", {}, {"na_reason": last.get("error", "unsupported SPARTA route")}
    if last.get("status") not in ("complete", "not_converged") or returncode != 0:
        return "failed", {"repeat_receipts": receipts}, {"na_reason": last.get("error", "missing SPARTA record")}
    retained = [r for r in records if r.get("retained")]
    complete = len(records) == warmup + repeats and len(retained) == repeats
    if last["status"] == "complete" and not complete:
        return "failed", {}, {"na_reason": "incomplete SPARTA repetitions"}
    chosen = (sorted(retained, key=lambda r: r["total_s"])[len(retained) // 2]
              if complete else last)
    metrics = {k: v for k, v in chosen.items() if k not in ("status", "retained", "repetition", "provenance")}
    metrics.update(sparta_provenance=chosen["provenance"], repeat_receipts=receipts,
                   retained_repeats=len(retained), warmup_repeats=warmup,
                   representative_repeat=chosen.get("repetition"))
    residuals = [r["rel_res"] for r in records]
    good = all(math.isfinite(x) and 0 <= x <= float(tol) for x in residuals)
    metrics["max_repeat_rel_res"] = max(residuals)
    return ("complete" if complete and good else "not_converged"), metrics, None


def run(operator, rhs, *, timeout, repeats, warmup, threads, tol, maxiter):
    ready, reason = available()
    if not ready:
        return "n/a", {}, {"na_reason": reason}
    with tempfile.TemporaryDirectory(prefix=".sparta-", dir=os.path.dirname(os.path.abspath(operator))) as work:
        command = [PYTHON, str(WORKER), os.path.abspath(operator), os.path.abspath(rhs), work,
                   "--repeat", str(repeats), "--warmup", str(warmup), "--threads", str(threads),
                   "--tol", str(tol), "--maxiter", str(maxiter)]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True) as process:
            timed_out = False
            try:
                _, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, stderr = process.communicate()
                timed_out = True
        try:
            records = [json.loads(p.read_text()) for p in sorted(Path(work).glob("rep*.json"),
                       key=lambda p: int(p.stem[3:]))]
            result = summarize(records, process.returncode, repeats, warmup, tol, timed_out)
        except (ValueError, KeyError, TypeError) as error:
            return "failed", {}, {"na_reason": f"invalid SPARTA receipt: {error}"}
        if timed_out:
            result[2]["process_deadline_s"] = timeout
            return result
        if not records:
            return "failed", {}, {"na_reason": stderr[-500:] or "SPARTA produced no records"}
        return result
