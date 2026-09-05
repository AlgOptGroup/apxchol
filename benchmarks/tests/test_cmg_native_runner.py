import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cmg_native_runner as cmg


class NativeCmgOutput(unittest.TestCase):
    def good(self):
        return cmg.parse_driver('CMG n=1024 nnz=4992 hierarchy_valid=1 levels=2 setup_flag=0 pcg_flag=0 iter=22 reported_relres=1e-9 true_relres=1e-9 adaptation_s=0.01 setup_s=0.2 solve_s=0.3 total_s=0.5 setup_calls=1 converged=1', 'CMGRSS 10240')

    def test_true_residual_wins_over_success_flag(self):
        values = self.good()
        self.assertEqual(cmg.classify(values, 0, 1e-9), 'complete')
        self.assertEqual(cmg.classify(values, 0, 1.01e-8), 'not_converged')
        self.assertEqual(cmg.classify(values, 0, float('nan')), 'not_converged')

    def test_runtime_and_unsupported_are_distinct(self):
        self.assertEqual(cmg.classify(None, 137), 'oom')
        self.assertEqual(cmg.classify(None, 2), 'failed')
        values = self.good()
        values['setup_flag'] = -1
        self.assertEqual(cmg.classify(values, 1), 'n/a')
        values['setup_flag'] = 2
        values['hierarchy_valid'] = 0
        self.assertEqual(cmg.classify(values, 1), 'n/a')

    def test_inconsistent_timer_is_rejected(self):
        with self.assertRaises(ValueError):
            cmg.parse_driver('CMG n=1024 nnz=4992 hierarchy_valid=1 levels=2 setup_flag=0 pcg_flag=0 iter=22 true_relres=1e-9 adaptation_s=0.01 setup_s=0.2 solve_s=0.3 total_s=0.8 setup_calls=1 converged=1', '')

if __name__ == '__main__':
    unittest.main()
