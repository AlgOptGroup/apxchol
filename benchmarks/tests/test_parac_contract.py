"""Regression boundaries found by the full Daint ParAC campaign audit."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parac_contract as contract
import parac_runner as parac
import runner_common as rc


def matrix(path, weight):
    path.write_text('%%MatrixMarket matrix coordinate real symmetric\n'
                    f'% fixture\n2 2 3\n1 1 2\n2 2 2\n2 1 {weight}\n')
    return str(path)


def run(rr='1e-9', **extra):
    return dict(factor='0.1', factor_setup='1.0', adapter='0.1', solve='100',
                iters='20', rr=rr, recur='1e-4', rhs_norm='1', n=2, nnz=4,
                rss_mb=1, backend='portable-cpp', returncode=0, **extra)


class PhysicsEligibility(unittest.TestCase):
    def test_rejects_even_tiny_positive_stored_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            for weight in ('1e-30', '0.25'):
                with self.subTest(weight=weight):
                    src = matrix(Path(tmp)/'input.mtx', weight)
                    with self.assertRaisesRegex(contract.UnsupportedOperator,
                                                '1 positive stored off-diagonal'):
                        contract.require_original_physics(src)
            for weight in ('0', '-0.25'):
                contract.require_original_physics(matrix(Path(tmp)/'input.mtx', weight))

    def test_malformed_input_is_failed_not_mathematically_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)/'input.mtx'
            for content in ('broken', '%%MatrixMarket matrix coordinate real general\n2 2 1\n'):
                src.write_text(content)
                with self.assertRaises(ValueError) as error:
                    contract.require_original_physics(src)
                self.assertNotIsInstance(error.exception, contract.UnsupportedOperator)

    def test_positive_physics_cannot_reach_cold_or_warm_cache_or_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = matrix(Path(tmp)/'input.mtx', '1e-30')
            for prepare in (parac._reorder_amd, parac._nnz_sort):
                for warm in (False, True):
                    with self.subTest(prepare=prepare.__name__, warm=warm), \
                         mock.patch.object(parac, '_prep_cache_valid', return_value=warm) as cache, \
                         mock.patch.object(parac, '_produce_upstream') as producer, \
                         mock.patch.object(parac, 'sh') as fallback:
                        with self.assertRaises(contract.UnsupportedOperator):
                            prepare('iter0010', src, 'op', augment=True)
                        cache.assert_not_called(); producer.assert_not_called(); fallback.assert_not_called()

    def test_cpu_and_gpu_stamp_unsupported_without_running_a_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = matrix(Path(tmp)/'input.mtx', '.2')
            for entry in (parac.run_cpu, parac.run_gpu):
                with self.subTest(entry=entry.__name__), \
                     mock.patch.object(parac, '_native_mtx', return_value=src), \
                     mock.patch.object(rc, 'PARAC_REORD', tmp), \
                     mock.patch.object(parac, '_prep_cache_valid', return_value=True), \
                     mock.patch.object(parac, '_run_once_cpu') as cpu, \
                     mock.patch.object(parac, '_run_once_gpu') as gpu, \
                     mock.patch.object(rc, 'emit_cell') as emit:
                    entry('iter0010')
                    self.assertEqual(len(emit.call_args_list), 2)
                    self.assertTrue(all(c.args[4] == 'n/a' for c in emit.call_args_list))
                    physics = next(c for c in emit.call_args_list if c.args[2]=='parac_physics')
                    self.assertIn('positive stored', physics.kwargs['matrix_meta']['parac_na_reason'])
                    cpu.assert_not_called(); gpu.assert_not_called()

    def test_graph_mode_still_accepts_positive_adjacency(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = matrix(Path(tmp)/'input.mtx', '.2')
            cached = Path(tmp)/'graph-amd.mtx'
            cached.write_text('cached')
            with mock.patch.object(rc, 'PARAC_REORD', tmp), \
                 mock.patch.object(parac, '_prep_cache_valid', return_value=True), \
                 mock.patch.object(parac, 'require_original_physics') as check:
                parac._reorder_amd('grid_500', src, 'graph', augment=False)
                check.assert_not_called()


class CalibrationBoundary(unittest.TestCase):
    def test_valid_cpu_formula_is_unchanged(self):
        probe = dict(iters='20', recur='1e-4', rr='1e-5', recurrence_reached='1')
        self.assertAlmostEqual(contract.calibrated_cpu_tolerance(probe, 1e-8, 2000),
                               (1e-8 * 1e-4 / 1e-5)**2)

    def test_invalid_or_capped_probe_preserves_evidence(self):
        good = dict(iters='20', recur='1e-4', rr='1e-5', recurrence_reached='1')
        for change in (dict(iters='2000'), dict(recurrence_reached='0'),
                       dict(rr='nan'), dict(rr='-1'), dict(recur=None), dict(returncode=1)):
            probe = {**good, **change}
            with self.subTest(change=change), self.assertRaises(contract.CalibrationFailed) as error:
                contract.calibrated_cpu_tolerance(probe, 1e-8, 2000)
            self.assertEqual(error.exception.probe, probe)

    def test_failed_cpu_calibration_runs_no_retained_repetitions(self):
        probe = run(); probe['iters'] = '2000'
        with mock.patch.object(rc, 'cell_done', return_value=False), \
             mock.patch.object(parac, '_run_once_cpu', return_value=probe) as driver, \
             mock.patch.object(rc, 'emit_cell') as emit:
            parac._measure_cpu('ipm', 'iter0010', 'input', 1, True, 'parac_physics')
            self.assertEqual(driver.call_count, 1)
            self.assertEqual(emit.call_args.args[4], 'failed')
            meta = emit.call_args.kwargs['matrix_meta']
            self.assertEqual(meta['retained_attempts'], 0)
            self.assertEqual(meta['parac_calibration_probe']['iters'], '2000')

    def test_failed_component_calibration_runs_no_retained_repetitions(self):
        probe = run(); probe['iters'] = '2000'
        with mock.patch.object(rc, 'cell_done', return_value=False), \
             mock.patch.object(parac, '_native_mtx', return_value=None), \
             mock.patch.object(parac, '_dump_component', return_value=('input',2,1,0)), \
             mock.patch.object(parac, '_reorder_amd', return_value=('amd',0,'upstream')), \
             mock.patch.object(parac, '_run_once_cpu', return_value=probe) as driver, \
             mock.patch.object(rc, 'emit_cell') as emit:
            parac._measure_cpu_graph_split('grids', 'grid_500')
            self.assertEqual(driver.call_count, 1)
            self.assertEqual(emit.call_args.args[4], 'failed')
            self.assertEqual(emit.call_args.kwargs['matrix_meta']['component_rank'], 0)

    def test_gpu_calibration_rejects_its_actual_300_iteration_cap(self):
        for probe in ({'iters':'300','rr':'1e-5'}, {'iters':'20','rr':'-nan'},
                      {'iters':'20','rr':'-1e-8'}, {'iters':'20','rr':'1e999'},
                      {'iters':'20','rr':'1e-9','returncode':1}, {'rr':'1e-9'}):
            with self.subTest(probe=probe), \
                 mock.patch.object(parac, '_run_once_gpu', return_value=probe), \
                 self.assertRaises(contract.CalibrationFailed):
                parac._calibrate_tol_gpu('driver', 'input')

    def test_number_parser_does_not_accept_malformed_numeric_prefix(self):
        for text in ('-nan', 'nan', 'inf', '-1e-8', '1e999'):
            self.assertEqual(parac._number_after('relative residual:',
                             'relative residual: '+text), text)
        for text in ('1e-8junk', '-', ''):
            self.assertIsNone(parac._number_after('relative residual:',
                              'relative residual: '+text))


class EveryRepetition(unittest.TestCase):
    def test_bad_nonmedian_cpu_run_rejects_cell(self):
        for residual in ('2e-8', '-1e-9', 'nan', 'inf'):
            runs = [run(), run(), run(residual)]
            for i,r in enumerate(runs): r['solve'] = str(100*(i+1))
            with self.subTest(residual=residual), \
                 mock.patch.object(rc,'cell_done',return_value=False), \
                 mock.patch.object(parac,'_calibrate_rel_tol',return_value=1e-16), \
                 mock.patch.object(parac,'_run_once_cpu',side_effect=runs), \
                 mock.patch.object(rc,'emit_cell') as emit:
                parac._measure_cpu('grids','grid_500','input',0,False,'parac')
                self.assertEqual(emit.call_args.args[4], 'not_converged')
                self.assertEqual(emit.call_args.args[5]['representative_repeat'], 2)
                self.assertEqual(emit.call_args.args[5]['rel_res'], 1e-9)

    def test_bad_nonmedian_component_run_rejects_global_cell(self):
        runs = [run(),run(),run('2e-8')]
        for i,r in enumerate(runs): r['solve'] = str(100*(i+1))
        with mock.patch.object(rc,'cell_done',return_value=False), \
             mock.patch.object(parac,'_native_mtx',return_value=None), \
             mock.patch.object(parac,'_dump_component',return_value=('input',2,1,0)), \
             mock.patch.object(parac,'_reorder_amd',return_value=('amd',0,'upstream')), \
             mock.patch.object(parac,'_calibrate_rel_tol',return_value=1e-16), \
             mock.patch.object(parac,'_run_once_cpu',side_effect=runs), \
             mock.patch.object(rc,'emit_cell') as emit:
            parac._measure_cpu_graph_split('grids','grid_500')
            self.assertEqual(emit.call_args.args[4],'not_converged')
            self.assertEqual(emit.call_args.args[5]['representative_repeat'],2)


class GpuExecutionBoundary(unittest.TestCase):
    def gpu_run(self, side_effect):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(parac, '_native_mtx', return_value='input'))
        stack.enter_context(mock.patch.object(parac, '_component_info', return_value=(1,2,0)))
        stack.enter_context(mock.patch.object(parac, '_prep_cache_valid', return_value=False))
        stack.enter_context(mock.patch.object(parac, '_nnz_sort', return_value=('sorted',0,'upstream')))
        stack.enter_context(mock.patch.object(parac.os.path, 'exists', return_value=True))
        stack.enter_context(mock.patch.object(rc, 'cell_done', return_value=False))
        stack.enter_context(mock.patch.object(parac, '_gpu_modes',
                           return_value=[('parac_graph','driver',{},None)]))
        driver=stack.enter_context(mock.patch.object(parac, '_run_once_gpu', side_effect=side_effect))
        emit=stack.enter_context(mock.patch.object(rc, 'emit_cell'))
        parac.run_gpu('grid_500')
        return driver, emit

    def test_failed_gpu_probe_launches_no_retained_calls(self):
        driver,emit=self.gpu_run([{'iters':'300','rr':'1e-5'}])
        self.assertEqual(driver.call_count,1)
        self.assertEqual(emit.call_args.args[4],'failed')
        self.assertEqual(emit.call_args.kwargs['matrix_meta']['retained_attempts'],0)

    def test_bad_nonmedian_gpu_run_rejects_cell(self):
        sample=dict(cuda_init='0.1',adapter='0.1',factor_setup='0.1',solver_setup='0.1',
                    factor='1',conv='1',spsv='1',solve='1',iters='20',rr='1e-9',n='2',nnz='4')
        runs=[{**sample,'solve_total':str(i),'rr':residual}
              for i,residual in enumerate(('1e-9','1e-9','2e-8'),1)]
        driver,emit=self.gpu_run([{'iters':'20','rr':'1e-5'}, *runs])
        self.assertEqual(driver.call_count,4)
        self.assertEqual(emit.call_args.args[4],'not_converged')
        self.assertEqual(emit.call_args.args[5]['representative_repeat'],2)
        self.assertEqual(emit.call_args.args[5]['rel_res'],1e-9)


class ThreadScalingBoundary(unittest.TestCase):
    def test_unsupported_and_failed_calibration_return_terminal_cells(self):
        import thread_scaling as scaling
        errors = [(contract.UnsupportedOperator('positive off-diagonal'), 'n/a'),
                  (contract.CalibrationFailed('capped', {'iters':'2000'}), 'failed')]
        for error, expected in errors:
            with self.subTest(error=error), \
                 mock.patch.object(scaling, '_prepare_parac', side_effect=error), \
                 mock.patch.object(parac, '_run_once_cpu') as driver:
                status, metrics = scaling.run_parac('grid_500', 16)
                self.assertEqual(status, expected)
                self.assertIsNone(metrics['total_s'])
                self.assertIn(str(error), metrics['parac_failure_reason'])
                driver.assert_not_called()

    def test_nonmedian_scaling_failure_is_not_complete(self):
        import thread_scaling as scaling
        runs = [run(), run(), run('2e-8')]
        for i,r in enumerate(runs): r['solve'] = str(100*(i+1))
        with mock.patch.object(scaling, '_prepare_parac', return_value=([('amd',False)],0)), \
             mock.patch.object(parac, '_calibrate_rel_tol', return_value=1e-16), \
             mock.patch.object(parac, '_run_once_cpu', side_effect=runs):
            status,metrics = scaling.run_parac('grid_500',16)
            self.assertEqual(status,'not_converged')
            self.assertEqual(metrics['representative_repeat'],2)
            self.assertEqual(metrics['rel_res'],1e-9)


class ThreadScalingEmission(unittest.TestCase):
    def test_failure_metrics_survive_outer_sweep_emission(self):
        import thread_scaling as scaling
        for status in ('n/a', 'failed'):
            metrics = {'total_s':None, 'parac_failure_reason':'probe/input rejected'}
            with self.subTest(status=status), ExitStack() as stack:
                for name,value in (('MATS',[('grid_500','grids','unused',False,True)]),
                                   ('CPP',[]),('THREADS',[16]),('ONLY_SERIES',set()),
                                   ('ONLY_MATRICES',set())):
                    stack.enter_context(mock.patch.object(scaling,name,value))
                stack.enter_context(mock.patch.object(scaling,'done',return_value=False))
                stack.enter_context(mock.patch.object(scaling,'run_parac',return_value=(status,metrics)))
                emit=stack.enter_context(mock.patch.object(scaling,'emit'))
                scaling.sweep()
                self.assertEqual(emit.call_count,1)
                self.assertEqual(emit.call_args.args[7],status)
                self.assertIn('parac_failure_reason',emit.call_args.args[6])


if __name__ == '__main__':
    unittest.main()
