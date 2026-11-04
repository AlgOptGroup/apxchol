"""SPARTA adapter checks without installing the optional native dependency."""
import json
import tempfile
import subprocess
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from scipy import sparse, io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util
_spec = importlib.util.spec_from_file_location("sparta_worker", Path(__file__).resolve().parents[1] / "sparta/worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)
import runner_common as rc
import sparta_runner
import sweep_fair


class Factor:
    def __init__(self, matrix, optimistic=False, backend="metal"):
        self.matrix = matrix.toarray()
        self.optimistic = optimistic
        self.backend = backend
        self.calls = []
        self.destroyed = False

    def factorize(self):
        return self

    def solve(self, rhs):
        self.calls.append(rhs.copy())
        out = np.linalg.solve(self.matrix, rhs)
        if self.optimistic and len(self.calls) == 1:
            out[-1] += 1e-4
        return out

    def diagnostics(self):
        return {"backend": self.backend, "options": {},
                "refinement_reports": [{"corrections": 0, "status": "converged",
                                        "error_estimate": float("nan")} ]}

    def destroy(self):
        self.destroyed = True


class SpartaTest(unittest.TestCase):
    def setUp(self):
        self.A = sparse.csc_matrix([[1., -1.], [-1., 1.]])
        self.b = np.array([1., -1.])

    def test_optimistic_grounded_stop_continues_using_same_factor(self):
        G, pins, _ = worker.ground(self.A, "laplacian")
        f = Factor(G, optimistic=True)
        first = worker.checked_solve(f, self.A, self.b, pins, 1e-8, 500)
        self.assertEqual(first['solve_passes'], 2)
        self.assertGreater(first['passes'][0]['original_rel_res'], 1e-8)
        self.assertLessEqual(first['rel_res'], 1e-8)
        self.assertEqual(first['factor_applications'], 2)
        # Another RHS reaches the same factor object, without factorization.
        second = worker.checked_solve(f, self.A, self.b * 3, pins, 1e-8, 500)
        self.assertLessEqual(second['rel_res'], 1e-8)
        self.assertEqual(len(f.calls), 3)
        json.dumps(first, allow_nan=False)

    def test_mixed_grounding_preserves_nonsingular_blocks(self):
        A = sparse.block_diag([self.A, [[2., -1.], [-1., 1.]], [[1e-100]], [[0.]]], format='csc')
        G, pins, count = worker.ground(A, "sddm")
        self.assertEqual(count, 4)
        self.assertEqual(pins.tolist(), [0, 5])
        np.testing.assert_array_equal(G.toarray()[2:5, 2:5], A.toarray()[2:5, 2:5])

    def test_budget_and_nonfinite_stop(self):
        G, pins, _ = worker.ground(self.A, "laplacian")
        f = Factor(G)
        r = worker.checked_solve(f, self.A, self.b, pins, 1e-8, 0)
        self.assertEqual(r['rel_res'], 1.)
        self.assertEqual(f.calls, [])
        f.solve = lambda _: np.array([np.nan, 0.])
        with self.assertRaises(FloatingPointError):
            worker.checked_solve(f, self.A, self.b, pins, 1e-8, 500)

    def test_no_cpu_fallback_and_factor_cleanup(self):
        G, _, _ = worker.ground(self.A, "laplacian")
        f = Factor(G, backend='cpu')
        module = mock.Mock(analyze=mock.Mock(return_value=f))
        with self.assertRaises(worker.Unavailable):
            worker.trial(module, self.A, self.b, 1e-8, 500, 8, "laplacian")
        self.assertTrue(f.destroyed)
        self.assertEqual(f.calls, [])

    def test_preparation_and_checks_are_timed(self):
        clock = [0.]
        original_ground = worker.ground
        def ground(A, matrix_class):
            clock[0] += 2
            return original_ground(A, matrix_class)
        class TimedFactor(Factor):
            def factorize(self):
                clock[0] += 5
                return self
            def solve(self, rhs):
                clock[0] += 7
                return super().solve(rhs)
        def analyze(G, **kwargs):
            clock[0] += 3
            return TimedFactor(G)
        def now():
            clock[0] += .01
            return clock[0]
        with mock.patch.object(worker, 'ground', side_effect=ground), mock.patch.object(worker.time, 'perf_counter', side_effect=now):
            r = worker.trial(mock.Mock(analyze=analyze), self.A, self.b, 1e-8, 500, 8, "laplacian")
        self.assertGreaterEqual(r['setup_s'], 10)
        self.assertGreaterEqual(r['solve_s'], 7)
        self.assertGreater(r['stop_check_s'], 0)
        self.assertEqual(r['total_s'], r['setup_s'] + r['solve_s'])

    def test_wrong_install_revision_unavailable(self):
        dist = mock.Mock()
        dist.read_text.return_value = json.dumps({'vcs_info': {'commit_id': 'wrong'}})
        with mock.patch.object(worker.platform, 'system', return_value='Darwin'), mock.patch.object(worker.importlib.metadata, 'distribution', return_value=dist):
            with self.assertRaises(worker.Unavailable):
                worker.runtime(mock.Mock(gpu_available=lambda: True))

    def test_sweep_optional_denominator(self):
        with mock.patch.object(sparta_runner, 'PYTHON', ''), mock.patch.object(sweep_fair, 'PARAC_ONLY', False):
            bare = sweep_fair.planned_cell_count([('a', {})], 'metal')
        with mock.patch.object(sparta_runner, 'PYTHON', '/venv/python'), mock.patch.object(sweep_fair, 'PARAC_ONLY', False):
            self.assertEqual(sweep_fair.planned_cell_count([('a', {})], 'metal'), bare + 1)

    def test_resume_requires_adapter_marker_and_metal_route(self):
        c = {'cell': {'solver': 'sparta', 'device': 'metal'}, 'status': 'complete',
             'metrics': {'stop_contract': 'original-v1', 'execution_route': 'metal'},
             'provenance': {'git_sha': 'abc', 'sparta_contract': rc.SPARTA_CONTRACT, 'sparta_revision': rc.SPARTA_REVISION}}
        self.assertTrue(rc.cell_meets_contract(c))
        c['provenance']['sparta_contract'] = 'sparta-metal-original-correction-v1'
        self.assertFalse(rc.cell_meets_contract(c))
        c['provenance']['sparta_contract'] = rc.SPARTA_CONTRACT
        c['metrics']['execution_route'] = 'cpu'
        self.assertFalse(rc.cell_meets_contract(c))
        c['metrics']['execution_route'] = 'metal'
        del c['provenance']['sparta_contract']
        self.assertFalse(rc.cell_meets_contract(c))

    @staticmethod
    def receipt(index, warmup=1):
        return dict(status="complete", repetition=index, retained=index >= warmup,
                    setup_s=1., solve_s=2., total_s=3., rel_res=1e-10,
                    stop_contract="original-v1", execution_route="metal", matrix_class="laplacian",
                    provenance={"sparta_revision": rc.SPARTA_REVISION,
                                "sparta_contract": rc.SPARTA_CONTRACT})

    def test_timeout_retains_completed_receipts(self):
        class Process:
            pid = 123456
            returncode = -9
            def __init__(self, command, **kwargs):
                self.work = Path(command[4]); self.calls = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def communicate(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    for i in range(2):
                        (self.work / f"rep{i}.json").write_text(json.dumps(SpartaTest.receipt(i)))
                    raise subprocess.TimeoutExpired('worker', timeout)
                return '', ''
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(sparta_runner, 'available', return_value=(True, '')), mock.patch.object(sparta_runner.subprocess, 'Popen', Process), mock.patch.object(sparta_runner.os, 'killpg') as kill:
            status, metrics, meta = sparta_runner.run(str(Path(directory) / 'A.mtx'), 'b.mtx', timeout=9, repeats=3, warmup=1, threads=8, tol=1e-8, maxiter=500, matrix_class="laplacian")
        self.assertEqual(status, 'timeout')
        self.assertEqual(len(metrics['partial_repetitions']), 2)
        self.assertEqual(len(metrics['repeat_receipts']), 2)
        self.assertEqual(metrics['sparta_provenance']['sparta_revision'], worker.REVISION)
        self.assertEqual(meta['timeout_scope'], 'logical_cell')
        self.assertIsNone(rc.timeout_cap({'timeout_cap_s': 9, 'matrix_meta': meta}))
        kill.assert_called_once()

    def test_summary_rejects_wrong_indexes_flags_and_source(self):
        for field, value in [('repetition', 1), ('retained', True), ('retained', 0)]:
            rows = [self.receipt(0), self.receipt(1)]
            rows[0][field] = value
            self.assertEqual(sparta_runner.summarize(rows, 0, 1, 1, 1e-8, "laplacian")[0], 'failed')
        rows = [self.receipt(0), self.receipt(1)]
        rows[1]['provenance']['sparta_revision'] = 'wrong'
        self.assertEqual(sparta_runner.summarize(rows, 0, 1, 1, 1e-8, "laplacian")[0], 'failed')
        rows = [self.receipt(0), self.receipt(1)]
        rows[1]['rel_res'] = 1e-4
        self.assertEqual(sparta_runner.summarize(rows, 0, 1, 1, 1e-8, "laplacian")[0], 'not_converged')

    def test_sweep_passes_shared_exports_and_keeps_logical_timeout(self):
        timeout = ('timeout', {'partial_repetitions': []}, {'timeout_scope': 'logical_cell'})
        seen = []
        def step(family, mid, solver, config, runner, label, **kwargs):
            if solver == 'sparta':
                seen.append(runner())
                return seen[-1][:2]
            return 'complete', {'total_s': 1.}
        with mock.patch.object(sweep_fair, 'DEVICE', 'metal'), mock.patch.object(sweep_fair, 'PARAC_ONLY', False), mock.patch.object(sweep_fair, 'APX_METAL', [('apxchol_v1', 'x')]), mock.patch.object(sweep_fair, 'step', side_effect=step), mock.patch.object(sweep_fair, 'cell_done', return_value=False), mock.patch.object(sweep_fair, 'dump_mtx', return_value='/shared/A.mtx'), mock.patch.object(sweep_fair, 'dump_rhs', return_value='/shared/b.mtx'), mock.patch.object(sparta_runner, 'PYTHON', '/env/python'), mock.patch.object(sparta_runner, 'run', return_value=timeout) as run:
            sweep_fair.do_matrix('grid_500', 'grids', 'grid', 500, True, 250000, None)
        self.assertEqual(run.call_args.args, ('/shared/A.mtx', '/shared/b.mtx'))
        self.assertEqual(run.call_args.kwargs['maxiter'], sweep_fair.MAXITER['metal'])
        self.assertEqual(seen, [timeout])

    def test_real_weighted_grid_export_uses_declared_class(self):
        A = sparse.csc_matrix(io.mmread(Path(__file__).with_name('fixtures') / 'sparta_weighted_grid4.mtx'))
        rows = A.tocsr()
        exact = [worker.math.fsum(rows.data[rows.indptr[i]:rows.indptr[i+1]]) for i in range(A.shape[0])]
        self.assertLess(min(exact), 0)  # The former blanket dominance guard refused this export.
        G, pins, count = worker.ground(A, 'laplacian')
        self.assertEqual(count, 1)
        self.assertEqual(pins.tolist(), [0])
        f = Factor(G)
        b = A @ np.arange(A.shape[0], dtype=float)
        self.assertLessEqual(worker.checked_solve(f, A, b, pins, 1e-8, 500)['rel_res'], 1e-8)

    def test_tiny_sddm_excess_is_retained_in_mixed_operator(self):
        shifted = self.A.toarray()
        shifted[0,0] = np.nextafter(1., 2.)
        A = sparse.block_diag([self.A, shifted], format='csc')
        G, pins, _ = worker.ground(A, 'sddm')
        self.assertEqual(pins.tolist(), [0])
        np.testing.assert_array_equal(G.toarray()[2:,2:], shifted)

    def test_invalid_laplacian_declaration_and_domain_refused(self):
        for A in [self.A + sparse.eye(2)*1e-6, self.A - sparse.eye(2)*1e-6]:
            with self.assertRaises(worker.Unavailable):
                worker.ground(A.tocsc(), 'laplacian')
        with self.assertRaises(worker.Unavailable):
            worker.ground(sparse.csc_matrix([[2., -1.], [0., 2.]]), 'sddm')

    def test_nondominant_spd_operator_is_not_reclassified(self):
        A = sparse.csc_matrix([[.9, -1.], [-1., 2.]])
        G, pins, _ = worker.ground(A, 'sddm')
        self.assertEqual(pins.tolist(), [])
        np.testing.assert_array_equal(G.toarray(), A.toarray())
        b = np.array([1., 2.])
        self.assertLessEqual(worker.checked_solve(Factor(G), A, b, pins, 1e-8, 500)['rel_res'], 1e-8)

    def test_positive_offdiagonal_spd_operator_is_preserved(self):
        A = sparse.csc_matrix([[2., 1.], [1., 2.]])
        self.assertGreater(np.linalg.eigvalsh(A.toarray()).min(), 0)
        G, pins, _ = worker.ground(A, 'sddm')
        self.assertEqual(pins.tolist(), [])
        np.testing.assert_array_equal(G.toarray(), A.toarray())
        b = np.array([1., -2.])
        self.assertLessEqual(worker.checked_solve(Factor(G), A, b, pins, 1e-8, 500)['rel_res'], 1e-8)

    def test_connected_and_mixed_signed_constant_null_components(self):
        signed = np.array([[4., 1., -5.], [1., 4., -5.], [-5., -5., 10.]])
        np.testing.assert_array_equal(signed.sum(axis=1), np.zeros(3))
        for A, expected in [(sparse.csc_matrix(signed), [0]),
                            (sparse.block_diag([self.A, signed, [[3.]]], format='csc'), [0, 2])]:
            original = A.toarray().copy()
            for matrix_class in ['sddm'] + (['laplacian'] if A.shape[0] == 3 else []):
                G, pins, _ = worker.ground(A, matrix_class)
                self.assertEqual(pins.tolist(), expected)
                b = A @ np.arange(A.shape[0], dtype=float)
                result = worker.checked_solve(Factor(G), A, b, pins, 1e-8, 500)
                self.assertLessEqual(result['rel_res'], 1e-8)
                np.testing.assert_array_equal(A.toarray(), original)

    def test_nonfinite_operator_is_rejected(self):
        for bad in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                worker.ground(sparse.csc_matrix([[bad]]), 'sddm')


if __name__ == '__main__':
    unittest.main()
