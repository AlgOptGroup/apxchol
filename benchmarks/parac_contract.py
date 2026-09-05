"""Input and calibration checks shared by the two ParAC benchmark runners."""
import math


class UnsupportedOperator(ValueError):
    """ParAC's sign normalization would change the original physics operator."""


def require_original_physics(path):
    """Reject positive stored off-diagonals before any preprocessing cache hit.

    Stream the coordinate input without a SciPy dependency or another matrix
    allocation. Count stored entries (a symmetric file stores only one triangle).
    This intentionally rejects positive entries even if duplicate cancellation
    could produce an M-matrix: that would require a different preparation rule.
    """
    with open(path) as source:
        header = source.readline().lower().split()
        if (len(header) != 5 or header[:3] != ["%%matrixmarket", "matrix", "coordinate"]
                or header[3] not in {"real", "integer"}
                or header[4] not in {"general", "symmetric"}):
            raise ValueError("physics eligibility requires real coordinate MatrixMarket input")
        entries = (line.split() for line in source
                   if line.strip() and not line.lstrip().startswith("%"))
        try:
            rows, cols, expected = map(int, next(entries))
        except (StopIteration, ValueError) as error:
            raise ValueError("invalid MatrixMarket dimensions") from error
        if rows <= 0 or rows != cols or expected < 0:
            raise ValueError("physics input must be a nonempty square matrix")
        positive = count = 0
        for fields in entries:
            if len(fields) != 3:
                raise ValueError("invalid MatrixMarket coordinate entry")
            row, col = int(fields[0]), int(fields[1])
            value = float(fields[2])
            if not (1 <= row <= rows and 1 <= col <= cols and math.isfinite(value)):
                raise ValueError("invalid or nonfinite MatrixMarket entry")
            count += 1
            positive += row != col and value > 0.0
        if count != expected:
            raise ValueError(f"MatrixMarket entry count differs: {count}/{expected}")
        if positive:
            raise UnsupportedOperator(
                f"unsupported original physics operator: {positive} positive stored "
                "off-diagonal entries; ParAC would replace them with -abs(weight)")


class CalibrationFailed(ValueError):
    def __init__(self, reason, probe):
        super().__init__(reason)
        self.probe = dict(probe)


def calibrated_cpu_tolerance(probe, tau, max_iter):
    """Keep ParAC's existing rescaling, only after a successful finite probe."""
    try:
        iterations = int(probe["iters"])
        recurrence, residual = float(probe["recur"]), float(probe["rr"])
        if probe.get("returncode", 0) != 0:
            raise ValueError("driver failed")
        if not 0 <= iterations < max_iter:
            raise ValueError("iteration limit reached")
        if probe.get("recurrence_reached") not in (None, True, 1, "1"):
            raise ValueError("recurrence tolerance was not reached")
        if not all(math.isfinite(x) and x > 0 for x in (recurrence, residual, tau)):
            raise ValueError("invalid residual statistics")
        tolerance = (tau * recurrence / residual) ** 2
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("invalid derived tolerance")
        return tolerance
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise CalibrationFailed(f"ParAC calibration failed: {error}", probe) from error
