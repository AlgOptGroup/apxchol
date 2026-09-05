import sys
from pathlib import Path
import unittest
import tempfile
import os
import shlex
import subprocess
from unittest import mock
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

    def test_logical_cell_deadline_is_not_single_solve_lower_bound(self):
        self.assertIsNone(cmg.rc.timeout_cap({'timeout_cap_s': 210,
            'matrix_meta': {'timeout_scope': 'logical_cell'}}))
        self.assertEqual(cmg.rc.timeout_cap({'timeout_cap_s': 210}), 210)

    def test_sweep_preserves_already_assembled_laplacian_dump(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.dict(os.environ, {'APXCHOL_CMG_NATIVE_RESULTS': directory}), \
             mock.patch.object(cmg.rc, 'cell_done', return_value=False), \
             mock.patch('cmg_matlab_runner._dump', return_value='/common/operator.mtx'), \
             mock.patch.object(cmg, 'prepare') as prepare, \
             mock.patch.object(cmg, 'run_prepared', return_value={'status': 'complete'}):
            for mid in ('ecology1', 'grid_500'):
                self.assertEqual(cmg.run_one(mid, 1), 'complete')
                self.assertEqual(prepare.call_args.args[0]['mode'], 'physics')

    def test_completed_repetitions_do_not_make_cell_deadline_a_solve_bound(self):
        import numpy as np
        import scipy.io
        now = [0.0]
        def run_command(command, **kwargs):
            now[0] += 70
            Path(shlex.split(command)[-1]).write_text('solution fixture')
            return subprocess.CompletedProcess(command, 0, '', '')
        metrics = self.good()
        metrics.update(setup_s=20.0, solve_s=50.0, total_s=70.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'driver'; binary.write_text('binary fixture')
            record = {'id':'fixture', 'family':'test'}
            prepared = (np.eye(2), np.ones(2), root/'operator', root/'rhs', {})
            with mock.patch.object(cmg.time, 'monotonic', side_effect=lambda:now[0]), \
                 mock.patch.object(cmg.rc, 'sh', side_effect=run_command), \
                 mock.patch.object(cmg.rc, 'git_sha', return_value='fixture'), \
                 mock.patch.object(cmg.rc, 'CELLS', str(root/'cells')), \
                 mock.patch.object(cmg, 'parse_driver', return_value=metrics), \
                 mock.patch.object(scipy.io, 'mmread', return_value=np.ones(2)):
                result = cmg.run_prepared(record, prepared, binary=binary, threads=1,
                    output=root/'run', repetitions=3, timeout=210)
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual(len(result['runs']), 3)  # warmup plus two completed repetitions
            import json
            cell = json.loads(next((root/'cells').rglob('*.json')).read_text())
            self.assertEqual(cell['metrics']['total_s'], 70)
            self.assertEqual(cell['timeout_cap_s'], 210)
            self.assertIsNone(cmg.rc.timeout_cap(cell))

if __name__ == '__main__':
    unittest.main()
