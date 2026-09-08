#!/usr/bin/env python3
"""Aggregate recorded eigenvalues, without running an eigensolver."""
import json
import math
from fractions import Fraction
from pathlib import Path


def main():
    data = json.loads(Path(__file__).with_name('tiny-spectra.json').read_text())
    assert len(data['laws']) == 15
    count = 0
    for law in data['laws']:
        rows = law['outcomes']
        probabilities = [Fraction(row['probability']) for row in rows]
        assert all(p > 0 for p in probabilities)
        assert sum(probabilities) == 1
        values = []
        for probability, row in zip(probabilities, rows):
            e = row['error_eigenvalues']
            assert all(math.isfinite(x) for x in e)
            assert min(e) > -1
            values.append((float(probability), sum(x*x for x in e),
                           max(abs(x) for x in e),
                           (1+max(e))/(1+min(e))))
        for column, key in enumerate(['J', 'mean_rho', 'mean_condition'], 1):
            value = math.fsum(row[0]*row[column] for row in values)
            assert math.isclose(value, law['metrics'][key], rel_tol=1e-12, abs_tol=1e-12)
        print(law['case'], law['method'], law['edge_budget'],
              '/'.join(f'{law["metrics"][key]:.6f}' for key in ['J', 'mean_rho', 'mean_condition']))
        count += len(rows)
    assert count == 522
    print('PASS: 15/15 saved laws, 522/522 outcomes; no eigensolver or spectral-optimum claim.')


if __name__ == '__main__':
    main()
