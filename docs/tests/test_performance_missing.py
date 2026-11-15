"""Missing measurements must not become failed runs or artificial speedups."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('matplotlib', 'numpy')), 'plot dependencies unavailable')
class MissingMeasurements(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('performance', Path(__file__).parents[1] / 'performance.py')
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)
        common = dict(matrix='grid', n=25, nnz=105, device='cpu', threads=4, variant='', **{'class': 'grids'})
        self.reference = dict(common, solver='apxchol_v1', config='bg+tree[vec_pool_aos]', status='complete', setup_s=1., solve_s=1., rel_res=1e-9, meets_tolerance=True)
        self.other = dict(common, solver='amgcl', config='', status='missing', setup_s=None, solve_s=None, rel_res=None, meets_tolerance=False)
        self.rows = [self.reference, self.other]
    def campaign(self):
        return self.p.Campaign({'tolerance': 1e-8, 'cells': self.rows})
    def test_missing_is_not_a_speedup(self):
        self.assertEqual(self.campaign().family_cell('cpu', 'AMGCL', 'grid'), (None, None, '—'))
    def test_actual_failure_is_retained(self):
        self.other['status'] = 'failed'
        self.assertEqual(self.campaign().family_cell('cpu', 'AMGCL', 'grid')[0], self.p.UNFINISHED)
    def test_pending_variant_prevents_all_failed_claim(self):
        self.rows.append(dict(self.other, config='other', status='failed'))
        self.assertIsNone(self.campaign().family_cell('cpu', 'AMGCL', 'grid')[0])
    def test_completed_result_is_still_compared(self):
        self.other.update(status='complete', setup_s=2., solve_s=2., rel_res=1e-9, meets_tolerance=True)
        self.assertEqual(self.campaign().family_cell('cpu', 'AMGCL', 'grid')[0], 2.)
    def test_empty_panels_are_labelled(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.p.family_chart(self.campaign(), 'test', output)
            self.p.best_other_chart(self.campaign(), 'test', output)
            self.assertIn('no comparable results', (output / 'test-families.svg').read_text())
            self.assertIn('No comparable results', (output / 'test-best-other.svg').read_text())
            self.assertNotIn('no measured competitor finished', (output / 'test-best-other.svg').read_text())
    def test_missing_reference_is_labelled(self):
        self.reference.update(status='missing', meets_tolerance=False)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.p.family_chart(self.campaign(), 'test', output)
            self.assertIn('no comparable results', (output / 'test-families.svg').read_text())
    def test_failure_legend_limits_claim_to_measurements(self):
        self.other['status'] = 'failed'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.p.best_other_chart(self.campaign(), 'test', output)
            self.assertIn('no measured competitor finished', (output / 'test-best-other.svg').read_text())

    def test_cap_is_not_fixed_speedup(self):
        self.other.update(status='exceeded_cap', cap_s=30.)
        run = self.campaign()
        self.assertEqual(run.family_cell('cpu', 'AMGCL', 'grid')[0], self.p.UNFINISHED)
        self.assertEqual(run.family_cell('cpu', 'AMGCL', 'grid')[2], 'cap')
        table = self.p.times_table(run, 'cpu')
        self.assertIn('cap', table)
        self.assertNotIn('≥20×', table)

    def test_distinct_failure_marks_are_preserved(self):
        for status, mark in [('timeout', 't/o'), ('failed', 'fail'), ('oom', 'oom')]:
            self.assertEqual(self.p.MARKS[status], mark)

if __name__ == '__main__':
    unittest.main()
