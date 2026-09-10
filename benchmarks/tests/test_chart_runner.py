#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import fair_charts as cpu  # noqa: E402
import chart_cells  # noqa: E402
import combined_charts as combined  # noqa: E402
sys.path.insert(0, str(BENCH / "daint"))
import render_campaign as daint_campaign  # noqa: E402
import fill_chart  # noqa: E402
import fill_pass  # noqa: E402
import gpu_charts as gpu  # noqa: E402
import runner_common as rc  # noqa: E402
import selector_levels  # noqa: E402
import stale_cells  # noqa: E402
import sweep_fair  # noqa: E402
import thread_scaling  # noqa: E402


def record(status, total, *, threads=16, cap=None, solver="amgcl", config=""):
    out = {
        "cell": {"family": "audit", "matrix_id": "m", "solver": solver,
                 "config": config, "threads": threads, "device": "cpu"},
        "metrics": {} if total is None else {"total_s": total},
        "status": status,
    }
    if cap is not None:
        out["timeout_cap_s"] = cap
    return out


class ShellHarnessTest(unittest.TestCase):
    @staticmethod
    def _popen():
        process = mock.Mock()
        process.communicate.return_value = ("", "")
        process.args = ""
        process.returncode = 0
        return process

    def test_core_dumps_are_disabled_by_default(self):
        process = self._popen()
        with mock.patch.object(rc.subprocess, "Popen", return_value=process) as popen:
            rc.sh("worker --flag", env={})
        self.assertEqual(popen.call_args.args[0], "ulimit -c 0; worker --flag")

    def test_core_dumps_can_be_enabled_for_diagnostics(self):
        process = self._popen()
        with mock.patch.object(rc.subprocess, "Popen", return_value=process) as popen:
            rc.sh("worker --flag", env={"APXCHOL_BENCH_COREDUMP": "1"})
        self.assertEqual(popen.call_args.args[0], "worker --flag")

    def test_core_and_memory_limits_compose(self):
        process = self._popen()
        with mock.patch.object(rc.subprocess, "Popen", return_value=process) as popen:
            rc.sh("worker", env={}, mem_cap_gb=2)
        self.assertEqual(
            popen.call_args.args[0],
            "ulimit -c 0; ulimit -v 2097152; worker",
        )


class GpuTimingIsolationTest(unittest.TestCase):
    def test_cpp_timing_does_not_start_memory_poller(self):
        output = (
            "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz\n"
            "AMGCL,m,2,4,1e-3,2e-3,3e-3,2,1e-9,1,750\n"
        )
        with mock.patch.object(sweep_fair, "DEVICE", "gpu"), \
             mock.patch.object(rc, "benchmark_openmp_env", return_value={}), \
             mock.patch.object(rc, "taskset_prefix", return_value="taskset -c 0"), \
             mock.patch.object(rc, "VramSampler", side_effect=AssertionError("timing polluted")), \
             mock.patch.object(sweep_fair, "sh", return_value=subprocess.CompletedProcess(
                 "benchmark", 0, output, "")):
            status, metrics = sweep_fair.run_cpp("", "amgcl_cuda", "", False)
        self.assertEqual(status, "complete")
        self.assertIs(metrics["gpu_memory_polling"], False)
        self.assertNotIn("max_vram_mb", metrics)

    def test_parac_shared_calibration_and_timing_path_has_no_memory_poller(self):
        import parac_runner
        with mock.patch.object(rc, "VramSampler", side_effect=AssertionError("timing polluted")), \
             mock.patch.object(parac_runner, "sh", return_value=subprocess.CompletedProcess(
                 "gpu_rchol", 0, "APX GPU solve phase time: 0.02\n", "")):
            metrics = parac_runner._run_once_gpu("gpu_rchol", "matrix.mtx", 1e-8)
        self.assertEqual(metrics["solve_total"], "0.02")
        self.assertIs(metrics["gpu_memory_polling"], False)
        self.assertNotIn("vram_mb", metrics)
        self.assertNotIn("max_vram_mb", metrics)


class JuliaDriverPathTest(unittest.TestCase):
    def test_julia_driver_and_project_are_root_derived(self):
        output = (
            "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz\n"
            "AC [Kyng16;Jl],m,2,4,1e-3,2e-3,3e-3,2,1e-9,1,750\n"
        )
        with mock.patch.object(rc, "taskset_prefix", return_value="taskset -c 0"), \
             mock.patch.object(sweep_fair.time, "monotonic",
                               side_effect=[100.0, 101.0, 102.0, 103.0]), \
             mock.patch.object(sweep_fair, "sh", return_value=subprocess.CompletedProcess(
                 "julia", 0, output, "")) as run:
            status, _metrics, _meta = sweep_fair.run_julia(
                "/tmp/operator.mtx", "ac", "laplacian", timeout=17)
        command = run.call_args.args[0]
        self.assertEqual(status, "complete")
        self.assertEqual(run.call_count, sweep_fair.REPS)
        self.assertEqual([call.kwargs["timeout"] for call in run.call_args_list],
                         [16, 15, 14])
        self.assertIn(f"--project={rc.ROOT}/benchmarks/julia", command)
        self.assertIn(f"{rc.ROOT}/benchmarks/julia/bench_laplacians.jl", command)


class SeriesRuleTest(unittest.TestCase):
    def test_gpu_thread_scope_does_not_mix_t16_with_t72(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(chart_cells, "_sha_contains", return_value=True):
            for threads in (16, 72):
                cell = record("complete", float(threads), threads=threads,
                              solver="amgcl_cuda")
                cell["cell"]["device"] = "gpu"
                (pathlib.Path(directory) / f"m_t{threads}__gpu.json").write_text(json.dumps(cell))
            rows = gpu.load(directory, threads=72)
            self.assertEqual(rows[("audit", "m")]["AMGCL (GPU)"]["total"], 72)

    def test_label_maps_are_injective(self):
        rc.require_injective_labels(cpu.LABELS, "CPU")
        rc.require_injective_labels(gpu.LABELS, "GPU")
        with self.assertRaises(RuntimeError):
            rc.require_injective_labels({("a", "x"): "same", ("a", "y"): "same"},
                                        "broken")

    def test_thread_choice_is_declared_not_status_ranked(self):
        complete_t1 = record("complete", 1.0, threads=1)
        timeout_t16 = record("timeout", None, threads=16, cap=60)
        self.assertIs(cpu._pick([complete_t1, timeout_t16], "audit", "m", "AMGCL"),
                      timeout_t16)

    def test_duplicate_same_series_is_rejected_not_ranked(self):
        with self.assertRaises(RuntimeError):
            cpu._pick([record("complete", 1.0), record("timeout", None, cap=60)],
                      "audit", "m", "AMGCL")

    def test_legacy_gpu_csv_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
            with self.assertRaises(ValueError):
                gpu.load(handle.name)

    def test_current_greedy_cell_wins_over_legacy_luby_regardless_of_order(self):
        legacy = record("complete", 1.0, solver="apxchol_v1",
                        config="luby+tree[vec_pool]")
        current = record("complete", 2.0, solver="apxchol_v1",
                         config="greedy+tree[vec_pool]")
        for records in ([legacy, current], [current, legacy]):
            data = cpu._cpu_selector_data(records, family="audit",
                                           storage="vec_pool")
            self.assertEqual(data[("greedy", "vec_pool")]["m"]["total_s"], 2.0)

    def test_terminal_chart_markers_keep_na_and_nonconvergence_distinct(self):
        self.assertEqual(combined._terminal_marker("n/a"), "n/a")
        self.assertEqual(combined._terminal_marker("not_converged"), "not_converged")
        self.assertEqual(combined._terminal_marker("failed"), "failed")
        self.assertIsNone(combined._terminal_marker("complete"))
        self.assertIsNone(combined._terminal_marker(None))

    def test_nonmemory_heatmaps_retain_declared_missing_and_na_rows(self):
        absent = dict(vals=[float("nan")], is_mem=False, timeouts=[False],
                      oom=[False], failed=[False], nconv=[False], na=[False])
        self.assertTrue(combined._keep_heatmap_row(**absent))
        self.assertTrue(combined._keep_heatmap_row(**dict(absent, na=[True])))
        self.assertFalse(combined._keep_heatmap_row(
            **dict(absent, is_mem=True, na=[True])))


class CurrentChartCellTest(unittest.TestCase):
    @staticmethod
    def _apx_record(sha, total):
        return {
            "cell": {"family": "audit", "matrix_id": "com-Amazon",
                     "solver": "apxchol_v1",
                     "config": rc.APXCHOL_DEFAULT_CONFIG,
                     "threads": 16, "device": "cpu"},
            "matrix_meta": {"kind": "graph"},
            "metrics": {"total_s": total},
            "status": "complete",
            "provenance": {"git_sha": sha},
        }

    def test_current_chart_filters_invalidated_cell_from_mixed_store(self):
        with tempfile.TemporaryDirectory() as store:
            root = pathlib.Path(store)
            (root / "old.json").write_text(json.dumps(self._apx_record("old", 1.0)))
            (root / "fresh.json").write_text(json.dumps(self._apx_record("fresh", 2.0)))
            with mock.patch.object(
                    chart_cells, "_sha_contains",
                    side_effect=lambda sha, _commit: sha == "fresh"):
                records = cpu.load(store)
        self.assertEqual([r["provenance"]["git_sha"] for r in records], ["fresh"])

    def test_exact_denominator_path_rejects_stale_cell(self):
        with tempfile.TemporaryDirectory() as store:
            path = pathlib.Path(store) / "old.json"
            path.write_text(json.dumps(self._apx_record("old", 1.0)))
            with mock.patch.object(chart_cells, "_sha_contains", return_value=False):
                with self.assertRaisesRegex(chart_cells.StaleCellError,
                                            "checked 1/1 selected cells"):
                    chart_cells.load_current_entries(
                        store, stale_policy="reject", announce=False)

    def test_timeout_without_persisted_cap_is_filtered_without_git_rule(self):
        timeout = record("timeout", None, solver="unscoped_solver")
        with tempfile.TemporaryDirectory() as store:
            (pathlib.Path(store) / "timeout.json").write_text(json.dumps(timeout))
            records, report = chart_cells.load_current_records(
                store, announce=False)
        self.assertEqual(records, [])
        self.assertEqual((report.selected, report.current, report.stale), (1, 0, 1))

    def test_fill_chart_rejects_legacy_records_without_provenance(self):
        legacy = {"family": "audit", "matrix_id": "m",
                  "solver": "apxchol_bg", "fill": 1.0}
        with tempfile.TemporaryDirectory() as store:
            (pathlib.Path(store) / "legacy.json").write_text(json.dumps(legacy))
            with self.assertRaisesRegex(chart_cells.CellStoreError,
                                        "rerun fill_pass.py"):
                fill_chart.load(store)

    def test_fill_chart_loads_current_schema_after_stale_gate(self):
        fill = {
            "schema": fill_pass.FILL_SCHEMA,
            "cell": {"family": "audit", "matrix_id": "com-Amazon",
                     "solver": "apxchol_v1", "config": "bg+tree[vec_pool]",
                     "threads": 16, "device": "cpu"},
            "matrix_meta": {"kind": "graph"},
            "metrics": {"fill": 1.25}, "status": "complete",
            "provenance": {"git_sha": "fresh"}, "series": "apxchol_bg",
        }
        with tempfile.TemporaryDirectory() as store:
            (pathlib.Path(store) / "fill.json").write_text(json.dumps(fill))
            with mock.patch.object(chart_cells, "_sha_contains", return_value=True):
                rows = fill_chart.load(store)
        self.assertEqual(rows[("audit", "com-Amazon")]["apxchol_bg"], 1.25)

    def test_fill_pass_emits_stale_cells_compatible_provenance(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(fill_pass, "OUT", store), \
             mock.patch.object(fill_pass, "git_sha", return_value="fresh"), \
             mock.patch.object(fill_pass.rc, "kind_of", return_value="graph"):
            fill_pass.emit("m", "audit", "apxchol_bg", 4, 8, 6)
            record = json.loads(
                (pathlib.Path(store) / "m__apxchol_bg.json").read_text())
        self.assertEqual(record["schema"], fill_pass.FILL_SCHEMA)
        self.assertEqual(record["cell"]["solver"], "apxchol_v1")
        self.assertEqual(record["cell"]["config"], "bg+tree[vec_pool]")
        self.assertEqual(record["provenance"]["git_sha"], "fresh")
        self.assertEqual(record["series"], "apxchol_bg")

    def test_selector_level_plot_only_rejects_legacy_csv_row(self):
        legacy = {"matrix_id": "grid_2000", "family": "g", "selector": "bg",
                  "fwd_lvls": "10", "bck_lvls": "10"}
        with self.assertRaisesRegex(chart_cells.CellStoreError,
                                    "rerun selector_levels.py"):
            selector_levels.validate_rows([legacy])

    def test_selector_level_current_row_passes_stale_gate(self):
        row = {"schema": selector_levels.LEVEL_SCHEMA,
               "matrix_id": "grid_2000", "family": "g", "selector": "bg",
               "config": "bg+tree[vec_pool]", "device": "cpu",
               "git_sha": "fresh", "fwd_lvls": "10", "bck_lvls": "10"}
        with mock.patch.object(chart_cells, "_sha_contains", return_value=True):
            selector_levels.validate_rows([row])


class ThreadScalingStoreTest(unittest.TestCase):
    def test_failed_no_csv_cell_does_not_abort_later_cells(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store), \
             mock.patch.object(thread_scaling, "MATS", [("m", "audit", "", False, False)]), \
             mock.patch.object(thread_scaling, "CPP", [("A", "solver", "cfg")]), \
             mock.patch.object(thread_scaling, "INCLUDE_PARAC", False), \
             mock.patch.object(thread_scaling, "THREADS", [1, 2]), \
             mock.patch.object(thread_scaling, "run_cpp", side_effect=[
                 ("failed", {"returncode": 1}), ("complete", {"total_s": 1.0})]), \
             mock.patch.object(thread_scaling, "benchmark_openmp_provenance", return_value={}):
            thread_scaling.sweep()
            statuses = [json.loads(path.read_text())["status"]
                        for path in sorted(pathlib.Path(store).glob("*.json"))]
            self.assertEqual(statuses, ["failed", "complete"])

    def test_cell_key_includes_solver_configuration(self):
        self.assertNotEqual(
            thread_scaling._cell_tag("apxchol_v1", "bg+tree[vec_pool]"),
            thread_scaling._cell_tag("apxchol_v1", "bg+tree[vec_pool_aos]"),
        )

    def test_scaling_timeout_has_exact_cap_and_old_schema_reruns(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store), \
             mock.patch.object(thread_scaling, "TIMEOUT", 123):
            thread_scaling.emit("m", "audit", "A", "solver", "cfg", 4,
                                None, "timeout")
            path = pathlib.Path(store) / "m__solver_cfg__t4.json"
            record = json.loads(path.read_text())
            self.assertEqual(record["schema"], thread_scaling.SCALING_SCHEMA)
            self.assertEqual(record["timeout_cap_s"], 123)
            self.assertIsNone(rc.timeout_cap(record))
            self.assertFalse(stale_cells.timeout_cap_is_stale(record))
            self.assertTrue(thread_scaling.done("m", "solver", "cfg", 4))

            record.pop("schema")
            path.write_text(json.dumps(record))
            self.assertFalse(thread_scaling.done("m", "solver", "cfg", 4))
            with self.assertRaises(RuntimeError):
                thread_scaling._scaling_records()

    def test_timeout_preserves_partial_repeat_receipts(self):
        receipt = b'BENCH_REPEAT phase=retained index=0 total_s=1.0\n'
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store), \
             mock.patch.object(thread_scaling, "sh", side_effect=subprocess.TimeoutExpired(
                 "benchmark", 5, output=b"partial output", stderr=receipt)):
            status, metrics = thread_scaling.run_cpp("", "amgcl", "", False, 1)
            self.assertEqual(status, "timeout")
            self.assertEqual(metrics["repeat_receipts"], [receipt.decode().strip()])
            self.assertEqual((pathlib.Path(store)/metrics["raw_stdout"]).read_text(), "partial output")

    def test_exact_scaling_denominator_is_validated(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store), \
             mock.patch.object(thread_scaling, "MATS", [("m", "audit", "", False, False)]), \
             mock.patch.object(thread_scaling, "CPP", [("A", "solver", "cfg")]), \
             mock.patch.object(thread_scaling, "THREADS", [1]):
            cell = {
                "schema": thread_scaling.SCALING_SCHEMA,
                "cell": {"matrix_id": "m", "family": "audit", "label": "A",
                         "solver": "solver", "config": "cfg", "threads": 1,
                         "device": "cpu"},
                "metrics": {"total_s": 1.0}, "status": "complete",
                "provenance": {"git_sha": thread_scaling.git_sha()},
            }
            path = pathlib.Path(store) / "m__solver_cfg__t1.json"
            path.write_text(json.dumps(cell))
            # ParAC is also part of the declared denominator.
            parac = json.loads(json.dumps(cell))
            parac["cell"].update(label="ParAC", solver="parac", config="")
            (pathlib.Path(store) / "m__parac_none__t1.json").write_text(json.dumps(parac))
            thread_scaling.validate_cells()
            (pathlib.Path(store) / "m__parac_none__t1.json").unlink()
            with self.assertRaises(RuntimeError):
                thread_scaling.validate_cells()

    def test_parac_scaling_uses_component_cache_route(self):
        thread_scaling._PARAC_PREPARED.clear()
        with mock.patch.object(thread_scaling.parac, "_uses_physics", return_value=False), \
             mock.patch.object(thread_scaling.parac, "_native_mtx", return_value=None), \
             mock.patch.object(thread_scaling.parac, "_dump_component",
                               side_effect=[("component.mtx", 10000, 2, 0.5),
                                            ("singleton.mtx", 1, 2, 0.1)]), \
             mock.patch.object(thread_scaling.parac, "_reorder_amd",
                               return_value=("component-amd.mtx", 0.25, "upstream")) as reorder:
            self.assertEqual(
                thread_scaling._prepare_parac("m"),
                ([("component-amd.mtx", False)], 0.85),
            )
            reorder.assert_called_once_with(
                "m", "component.mtx", "comp0", deadline=None)
        thread_scaling._PARAC_PREPARED.clear()

    def test_parac_scaling_uses_native_connected_file_route(self):
        thread_scaling._PARAC_PREPARED.clear()
        with mock.patch.object(thread_scaling.parac, "_uses_physics", return_value=False), \
             mock.patch.object(thread_scaling.parac, "_native_mtx",
                               return_value="native.mtx"), \
             mock.patch.object(thread_scaling.parac, "_component_info",
                               return_value=(1, 10000, 0.2)), \
             mock.patch.object(thread_scaling.parac, "_dump_component") as dump, \
             mock.patch.object(thread_scaling.parac, "_reorder_amd",
                               return_value=("native-amd.mtx", 0.25, "upstream")) as reorder:
            self.assertEqual(
                thread_scaling._prepare_parac("m"),
                ([("native-amd.mtx", False)], 0.45),
            )
            dump.assert_not_called()
            reorder.assert_called_once_with(
                "m", "native.mtx", "native-comp0", deadline=None)
        thread_scaling._PARAC_PREPARED.clear()


class CellSchemaTest(unittest.TestCase):
    def test_cuda_init_is_parsed_as_separate_seconds(self):
        stderr = ("noise\n[bench] cuda_init (once, before any timed solver): "
                  "629.6 ms\nmore noise\n")
        self.assertAlmostEqual(rc.parse_cuda_init(stderr), 0.6296)
        self.assertIsNone(rc.parse_cuda_init("no CUDA build"))

    def test_timeout_requires_and_persists_exact_cap(self):
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            with self.assertRaises(ValueError):
                rc.emit_cell("audit", "m", "solver", "", "timeout", {}, 16, "cpu", {})
            path = rc.emit_cell("audit", "m", "solver", "", "timeout", {}, 16,
                                "cpu", {}, timeout_cap_s=73.5)
            with open(path) as handle:
                cell = json.load(handle)
            self.assertEqual(cell["schema"], 2)
            self.assertEqual(cell["timeout_cap_s"], 73.5)
            self.assertEqual(rc.timeout_cap(cell), 73.5)

    def test_non_timeout_does_not_accept_cap(self):
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            with self.assertRaises(ValueError):
                rc.emit_cell("audit", "m", "solver", "", "complete", {}, 16, "cpu",
                             {}, timeout_cap_s=60)

    def test_old_timeout_is_stale_and_not_terminal_for_sweep(self):
        old = record("timeout", None)
        self.assertTrue(stale_cells.timeout_cap_is_stale(old))
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            path = pathlib.Path(rc.cell_path("audit", "m", "amgcl", "", 16, "cpu"))
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(old))
            with mock.patch.object(sweep_fair, "DEVICE", "cpu"):
                self.assertFalse(sweep_fair.cell_done("audit", "m", "amgcl", ""))

    def test_accounting_stale_rules_are_device_and_solver_scoped(self):
        def commits(solver, matrix, device):
            return {commit for commit, _reason in stale_cells.matching_rules(
                solver, matrix, "graph", device)}

        self.assertIn("5a18d14", commits("amgcl_cuda", "com-Amazon", "gpu"))
        self.assertNotIn("5a18d14", commits("amgcl", "com-Amazon", "cpu"))
        self.assertIn("a4938af", commits("apxchol_v1", "com-Amazon", "cpu"))
        self.assertIn("a4938af", commits("parac", "com-Amazon", "cpu"))
        self.assertIn("a4938af", commits("amgcl", "as-Skitter", "cpu"))
        self.assertNotIn("a4938af", commits("amgcl", "com-Amazon", "cpu"))
        self.assertNotIn("a4938af", commits("cmg", "as-Skitter", "cpu"))

    def test_new_timeout_is_terminal_for_sweep(self):
        new = record("timeout", None, cap=60)
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            path = pathlib.Path(rc.cell_path("audit", "m", "amgcl", "", 16, "cpu"))
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(new))
            with mock.patch.object(sweep_fair, "DEVICE", "cpu"):
                self.assertTrue(sweep_fair.cell_done("audit", "m", "amgcl", ""))


class CapReferenceTest(unittest.TestCase):
    def test_gpu_cap_reference_uses_declared_default_not_fastest_selector(self):
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            base = dict(family="audit", mid="m", solver="apxchol_v1", status="complete",
                        threads=16, device="gpu", prov={"git_sha": rc.git_sha()})
            rc.emit_cell(config=sweep_fair.APX_DEFAULT_CONFIG,
                         metrics={"total_s": 10.0}, **base)
            rc.emit_cell(config="greedy+tree[vec_pool]", metrics={"total_s": 1.0}, **base)
            self.assertEqual(sweep_fair.gpu_apx_total("audit", "m"), 10.0)

    def test_gpu_cap_reference_uses_campaign_thread_count(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(sweep_fair, "THREADS", 72):
            rc.emit_cell("audit", "m", "apxchol_v1",
                         sweep_fair.APX_DEFAULT_CONFIG, "complete",
                         {"total_s": 7.2}, 72, "gpu", {"git_sha": rc.git_sha()})
            self.assertEqual(sweep_fair.gpu_apx_total("audit", "m"), 7.2)


class FairSweepSelectionTest(unittest.TestCase):
    def test_full_fair_plan_denominator_includes_indexed_gpu_bg_ablation(self):
        selected = list(sweep_fair.selected_matrices(
            {"grids", "suitesparse", "ipm"}))
        self.assertEqual(sweep_fair.APX_GPU[0],
                         ("apxchol_v1", sweep_fair.APX_DEFAULT_CONFIG))
        self.assertIn(("apxchol_v1", "bg+tree[vec_pool]"), sweep_fair.APX_GPU[1:])
        self.assertEqual(sweep_fair.planned_cell_count(selected, "cpu"), 615)
        self.assertEqual(sweep_fair.planned_cell_count(selected, "gpu"), 243)
        self.assertEqual(
            sweep_fair.planned_cell_count(selected, "cpu")
            + sweep_fair.planned_cell_count(selected, "gpu"),
            858,
        )

    def test_orkut_size_gate_always_keeps_declared_default(self):
        configs = [
            ("apxchol_v1", sweep_fair.APX_DEFAULT_CONFIG),
            ("apxchol_v1", "bg+tree[vec_pool]"),
            ("apxchol_v1", "bg+tree[vec]"),
            ("apxchol_v1", "bg+tree[bstr]"),
        ]
        with mock.patch.object(sweep_fair, "APX", configs):
            self.assertEqual(
                sweep_fair.cpu_apx_configs_for("com-Orkut"),
                configs[:3],
            )
            self.assertIs(sweep_fair.cpu_apx_configs_for("iter0040"), configs)

    def test_all_registry_entries_are_selected_once(self):
        selected = list(sweep_fair.selected_matrices(
            {"grids", "suitesparse", "ipm"}))
        self.assertEqual(len(selected), len(rc.MATRICES))
        self.assertEqual([mid for mid, _ in selected], list(rc.MATRICES))

    def test_file_backed_families_survive_registry_schema(self):
        selected = dict(sweep_fair.selected_matrices(
            {"suitesparse", "ipm"}, {"parabolic_fem", "iter0010"}))
        self.assertEqual(set(selected), {"parabolic_fem", "iter0010"})
        self.assertEqual(selected["parabolic_fem"]["cls"], "sddm")
        self.assertEqual(selected["iter0010"]["cls"], "sddm")
        self.assertTrue(pathlib.Path(selected["parabolic_fem"]["spec"]).is_absolute())

    def test_main_dispatches_suitesparse_and_ipm_without_unpacking_tuples(self):
        argv = ["sweep_fair.py", "--families", "suitesparse,ipm", "--only",
                "parabolic_fem,iter0010", "--no-julia", "--no-parac", "--no-cmg"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(sweep_fair, "do_matrix") as do_matrix, \
             mock.patch.object(sweep_fair, "JULIA", []), \
             mock.patch.object(sweep_fair, "RUN_PARAC", False), \
             mock.patch.object(sweep_fair, "RUN_CMG", False), \
             mock.patch("builtins.print"):
            sweep_fair.main()

        dispatched = {call.args[0]: call.args[1:] for call in do_matrix.call_args_list}
        self.assertEqual(set(dispatched), {"parabolic_fem", "iter0010"})
        self.assertEqual(dispatched["parabolic_fem"][0], "suitesparse")
        self.assertEqual(dispatched["iter0010"][0], "ipm")


class JuliaPreflightTest(unittest.TestCase):
    def test_missing_ignored_manifest_fails_before_cells_run(self):
        with mock.patch.object(sweep_fair.os.path, "isfile", return_value=False):
            ready, reason = sweep_fair.julia_preflight()
        self.assertFalse(ready)
        self.assertIn("Manifest.toml", reason)
        self.assertIn("Pkg.instantiate", reason)

    def test_package_load_failure_keeps_useful_stderr(self):
        failed = __import__("subprocess").CompletedProcess(
            "julia", 1, "", "root cause\nin expression starting at driver.jl:27\n")
        with mock.patch.object(sweep_fair.os.path, "isfile", return_value=True), \
             mock.patch.object(sweep_fair.shutil, "which", return_value="/bin/julia"), \
             mock.patch.object(sweep_fair, "sh", return_value=failed):
            ready, reason = sweep_fair.julia_preflight()
        self.assertFalse(ready)
        self.assertIn("root cause", reason)


class DaintCampaignRendererTest(unittest.TestCase):
    def test_committed_csv_extracts_are_typed_reproduction_inputs(self):
        scaling, baseline, pivots = daint_campaign.load_csv_inputs(BENCH / "daint")
        self.assertEqual(len(scaling), daint_campaign.SCALING_RECORDS)
        self.assertEqual(len(baseline), daint_campaign.BASELINE_RECORDS)
        self.assertEqual(len(pivots), daint_campaign.PIVOT_RECORDS)
        self.assertIsInstance(scaling[0]["threads"], int)
        self.assertIsInstance(scaling[0]["setup_ms"], float)
        self.assertIsInstance(baseline[0]["seed"], int)
        self.assertIsInstance(pivots[0]["worst_parallel_lpt"], float)

    def test_scaling_validator_enforces_189_record_grid_and_timing_identity(self):
        scaling, _baseline, _pivots = daint_campaign.load_csv_inputs(BENCH / "daint")
        max_delta = daint_campaign.validate_scaling_records(scaling)
        self.assertLessEqual(max_delta, daint_campaign.SCALING_TOTAL_ABS_TOL_MS)

        with self.assertRaisesRegex(RuntimeError, "checked 188/189"):
            daint_campaign.validate_scaling_records(scaling[:-1])

        inconsistent = [dict(row) for row in scaling]
        inconsistent[0]["total_ms"] += 0.01
        with self.assertRaisesRegex(RuntimeError, r"setup\+pcg=total"):
            daint_campaign.validate_scaling_records(inconsistent)

    def test_scaling_fixture_renders_three_panel_png(self):
        scaling, _baseline, _pivots = daint_campaign.load_csv_inputs(BENCH / "daint")
        daint_campaign.validate_scaling_records(scaling)
        medians = daint_campaign.scaling_medians(scaling)
        self.assertEqual(
            daint_campaign.SCALING_PANELS,
            (("total_ms", "Total"), ("pcg_ms", "PCG / solve"),
             ("setup_ms", "Setup")),
        )
        with tempfile.TemporaryDirectory() as output, \
             mock.patch.object(daint_campaign.plt, "subplots",
                               wraps=daint_campaign.plt.subplots) as subplots:
            path = pathlib.Path(output) / "scaling.png"
            daint_campaign.render_scaling_figure(path, medians)
            self.assertEqual(subplots.call_args.args, (1, 3))
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 10_000)

    def test_incomplete_scaling_fails_before_rendering(self):
        scaling, baseline, pivots = daint_campaign.load_csv_inputs(BENCH / "daint")
        with tempfile.TemporaryDirectory() as output, \
             mock.patch.object(daint_campaign, "load_csv_inputs",
                               return_value=(scaling[:-1], baseline, pivots)), \
             mock.patch.object(daint_campaign, "render_scaling_figure") as scaling_fig, \
             mock.patch.object(daint_campaign, "render_baseline_figure") as baseline_fig, \
             mock.patch.object(daint_campaign, "render_breakdown_figure") as breakdown_fig:
            with self.assertRaisesRegex(RuntimeError, "checked 188/189"):
                daint_campaign.main([
                    "--csv-input", str(BENCH / "daint"), "--output", output,
                ])
        scaling_fig.assert_not_called()
        baseline_fig.assert_not_called()
        breakdown_fig.assert_not_called()

    def test_csv_reproduction_bypasses_raw_logs_and_does_not_rewrite_extracts(self):
        with tempfile.TemporaryDirectory() as output, \
             mock.patch.object(daint_campaign, "parse_scaling") as parse_scaling, \
             mock.patch.object(daint_campaign, "parse_baseline") as parse_baseline, \
             mock.patch.object(daint_campaign, "parse_structure") as parse_structure, \
             mock.patch.object(daint_campaign, "write_csv") as write_csv, \
             mock.patch.object(daint_campaign, "render_scaling_figure") as scaling_fig, \
             mock.patch.object(daint_campaign, "render_baseline_figure") as baseline_fig, \
             mock.patch.object(daint_campaign, "render_breakdown_figure") as breakdown_fig, \
             mock.patch.object(daint_campaign, "write_summary") as write_summary:
            daint_campaign.main([
                "--csv-input", str(BENCH / "daint"), "--output", output,
            ])
        parse_scaling.assert_not_called()
        parse_baseline.assert_not_called()
        parse_structure.assert_not_called()
        write_csv.assert_not_called()
        scaling_fig.assert_called_once()
        baseline_fig.assert_called_once()
        breakdown_fig.assert_called_once()
        write_summary.assert_called_once()


class HeatmapColorScaleTest(unittest.TestCase):
    def test_log_ramp_and_ticks_cover_small_and_large_observed_ratios(self):
        from matplotlib.colors import LogNorm
        from matplotlib.figure import Figure
        captured = []
        def capture(fig, filename, **kwargs):
            image = fig.axes[0].images[0]
            captured.append((image.norm, image.colorbar.get_ticks()))
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(Figure, "savefig", capture):
            for largest in [1.0, 1.4, 1.6, 360.0]:
                gpu.value_heatmap(["grid_500"], ["fast", "slow"], [[1], [largest]],
                                  pathlib.Path(directory)/"test.png", title="TEST ONLY")
        for largest, (norm, ticks) in zip([1.0, 1.4, 1.6, 360.0], captured):
            self.assertIsInstance(norm, LogNorm)
            self.assertEqual(norm.vmin, 1.0)
            self.assertGreaterEqual(norm.vmax, largest)
            self.assertTrue(all(norm.vmin <= tick <= norm.vmax for tick in ticks))
            self.assertEqual(ticks[0], norm.vmin)
            self.assertEqual(ticks[-1], norm.vmax)
        norm = captured[-1][0]
        self.assertEqual(norm.vmax, 360.0)
        self.assertLess(norm(16.0), norm(100.0))
        self.assertLess(norm(100.0), norm(360.0))


class CompactScalingPlotTest(unittest.TestCase):
    def test_gap_and_measured_reference_survive_log_axes(self):
        import math
        from matplotlib.figure import Figure
        records = []
        for threads, status, value in [(1, "complete", 10), (2, "failed", None),
                                       (4, "complete", 4)]:
            row = record(status, value, threads=threads, solver="solver")
            row["cell"]["label"] = "A"
            if value is not None:
                row["metrics"].update(setup_s=value, solve_s=value)
            row["provenance"] = {"git_sha": "fresh"}
            if threads == 4:
                row["provenance"]["scaling_baseline"] = {
                    "setup_s": 12, "solve_s": 12}
            records.append((str(threads), row))
        captured = {}
        def capture(fig, filename, **kwargs):
            ax = fig.axes[0]
            captured[pathlib.Path(filename).name] = (
                list(ax.lines[0].get_ydata()), ax.get_xscale(), ax.get_yscale(),
                [line.get_label() for line in ax.lines])
        with tempfile.TemporaryDirectory() as out, \
             mock.patch.object(thread_scaling, "_scaling_records", return_value=records), \
             mock.patch.object(thread_scaling, "MATS", [("m", "audit", "", False, False)]), \
             mock.patch.object(thread_scaling, "CPP", [("A", "solver", "")]), \
             mock.patch.object(thread_scaling, "THREADS", [1, 2, 4]), \
             mock.patch.object(Figure, "savefig", capture):
            thread_scaling.charts(out, compact=True)
        self.assertEqual(len(captured), 4)
        for name, (values, xscale, yscale, labels) in captured.items():
            self.assertEqual((xscale, yscale), ("log", "log"))
            self.assertTrue(math.isnan(values[1]))
            self.assertEqual(values[2], 3 if "speedup" in name else 4)
            self.assertEqual("ideal" in labels, "speedup" in name)

    def test_phase_control_gap_preserves_raw_export_and_unflagged_phases(self):
        import csv
        import math
        from matplotlib.figure import Figure
        records = []
        for threads, value in [(1, 10), (2, 8), (4, 5)]:
            row = record("complete", 2 * value, threads=threads, solver="solver")
            row["cell"]["label"] = "A"
            row["metrics"].update(setup_s=value, solve_s=value)
            row["provenance"] = {"git_sha": "fresh"}
            if threads == 2:
                row["provenance"]["valid_ratios"] = {"solve_s": {"valid": False}}
            records.append((str(threads), row))
        captured = {}
        def capture(fig, filename, **kwargs):
            captured[pathlib.Path(filename).name] = list(fig.axes[0].lines[0].get_ydata())
        with tempfile.TemporaryDirectory() as out, \
             mock.patch.object(thread_scaling, "_scaling_records", return_value=records), \
             mock.patch.object(thread_scaling, "MATS", [("m", "audit", "", False, False)]), \
             mock.patch.object(thread_scaling, "CPP", [("A", "solver", "")]), \
             mock.patch.object(thread_scaling, "THREADS", [1, 2, 4]), \
             mock.patch.object(Figure, "savefig", capture):
            thread_scaling.charts(out, compact=True)
            destination = pathlib.Path(out) / "raw.csv"
            thread_scaling.export_csv(str(destination))
            with destination.open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(captured), 4)
        for name, values in captured.items():
            self.assertEqual(math.isnan(values[1]), "solve" in name)
            self.assertEqual(values[0], 1 if "speedup" in name else 10)
            self.assertEqual(values[2], 2 if "speedup" in name else 5)
            if "setup" in name:
                self.assertEqual(values[1], 1.25 if "speedup" in name else 8)
        self.assertEqual(rows[1]["solve_s"], "8")
        self.assertEqual(rows[1]["solve_ratio_valid"], "False")
        self.assertEqual(rows[1]["setup_ratio_valid"], "")
        self.assertEqual(records[1][1]["metrics"]["solve_s"], 8)

    def test_compact_rejects_more_than_three_panels(self):
        with mock.patch.object(thread_scaling, "_scaling_records", return_value=[]), \
             mock.patch.object(thread_scaling, "MATS", [(str(i),) for i in range(4)]):
            with self.assertRaisesRegex(ValueError, "at most three"):
                thread_scaling.charts(compact=True)


class SamplerProfileTest(unittest.TestCase):
    def setUp(self):
        from contextlib import ExitStack
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        for module, names in [(cpu, ("ORDER", "APX_SERIES", "POSTER_SOLVERS", "COLORS")),
                              (gpu, ("ORDER", "APX_SERIES", "APX_DEFAULT", "COLORS")),
                              (combined, ("SOLVERS", "ENC"))]:
            for name in names:
                value = getattr(module, name)
                self.context.enter_context(mock.patch.object(
                    module, name, value.copy() if isinstance(value, (list, dict)) else value))

    @staticmethod
    def series():
        return {(device, solver, config)
                for device, module in [("cpu", cpu), ("gpu", gpu)]
                for (solver, config), label in module.LABELS.items() if label in module.ORDER}

    def test_twenty_declared_rows_keep_packed_cmg_and_fifteen_competitors(self):
        historical = self.series()
        self.assertEqual(len(historical), 18)
        combined.select_sampler_comparison()
        current = self.series()
        self.assertEqual(len(current), 20)
        self.assertEqual(len(rc.MATRICES), 27)
        self.assertEqual({row for row in historical if row[1] not in {"apxchol_v1", "cmg"}},
                         {row for row in current if row[1] != "apxchol_v1"})
        self.assertNotIn(("gpu", "apxchol_v1", "bg+tree[vec_pool_aos]"), current)
        self.assertIn(("gpu", "apxchol_v1", "bg+tree/gpu-owned-q08[vec_pool_aos]"), current)
        self.assertIn(("gpu", "apxchol_v1", "bg+tree/gpu-owned-q02[vec_pool_aos]"), current)
        self.assertIn(("gpu", "apxchol_v1", "bg+trace_cycle/gpu-owned-q08[vec_pool_aos]"), current)
        self.assertEqual(gpu.LABELS[("apxchol_v1", "bg+tree/gpu-owned-q02[vec_pool_aos]")],
                         "apxchol/GKS q=0.2 (GPU)")
        self.assertEqual(gpu.LABELS[("apxchol_v1", "bg+tree/gpu-owned-q08[vec_pool_aos]")],
                         "apxchol/GKS q=0.8 (GPU)")
        q08 = next(row for row in combined.SOLVERS if row[0] == "apxchol/GKS q=0.8")
        self.assertIsNone(q08[2])
        self.assertEqual(q08[3], "apxchol/GKS q=0.8 (GPU)")
        self.assertNotIn(("cpu", "cmg", ""), current)
        self.assertIn(("cpu", "cmg_packed", "original-operator"), current)
        self.assertIn(("cmg", ""), cpu.LABELS)
        self.assertNotIn("CMG (MATLAB)†", cpu.POSTER_SOLVERS)
        self.assertNotIn("CMG (MATLAB)†", {row[0] for row in combined.SOLVERS})
        self.assertNotIn(("cpu", "apxchol_v1", "bg+heavy_core_k2[vec_pool_aos]"), current)
        rc.require_injective_labels(cpu.LABELS, "CPU")
        rc.require_injective_labels(gpu.LABELS, "GPU")

    def test_gpu_trace_fill_is_distinct_from_both_gks_quantiles(self):
        rows = {("grids", "grid_500"): {"apxchol_gpu_q08": 2.0,
                "apxchol_gpu_q02": 2.5, "apxchol_gpu_trace_q08": 3.0}}
        with mock.patch.object(gpu, "value_heatmap") as plot:
            self.assertTrue(fill_chart.heatmap(rows, "grids", "TEST-ONLY.png"))
        labels, values = plot.call_args.args[1:3]
        self.assertEqual(len(labels), 13)
        for label, value in [("apxchol GPU GKS q=0.8", 2.0),
                             ("apxchol GPU GKS q=0.2", 2.5),
                             ("apxchol GPU trace-cycle q=0.8", 3.0)]:
            self.assertEqual(values[labels.index(label), 0], value)

    def test_sequential_sampler_then_historical_render_restores_profile(self):
        import render_snapshot
        r = record("complete", 2, threads=72, solver="apxchol_v1")
        r["cell"].update(matrix_id=next(iter(rc.MATRICES)), config="bg+tree[vec_pool_aos]")
        historical = self.series()
        old_solvers, old_encoding = list(combined.SOLVERS), combined.ENC
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(chart_cells, "load_current_records", return_value=([r], {})), \
             mock.patch.object(render_snapshot.subprocess, "run"):
            root = pathlib.Path(directory)
            render_snapshot.render(root, root / "samplers", 72, "TEST ONLY", sampler_comparison=True)
            render_snapshot.render(root, root / "historical", 72, "TEST ONLY", sampler_comparison=False)
            current = json.loads((root / "samplers/coverage.json").read_text())
            restored = json.loads((root / "historical/coverage.json").read_text())
        self.assertEqual((current["series_profile"], current["expected_headline_cells"]),
                         ("sampler-comparison", 540))
        self.assertEqual((restored["series_profile"], restored["expected_headline_cells"]),
                         ("historical-default", 486))
        self.assertEqual(self.series(), historical)
        self.assertEqual(gpu.APX_DEFAULT, "apxchol/bg (GPU)")
        combined.select_sampler_comparison(True)
        combined.select_sampler_comparison(True)
        self.assertEqual(combined.ENC.count("apxchol GPU bars"), 1)
        combined.select_sampler_comparison(False)
        self.assertEqual(combined.SOLVERS, old_solvers)
        self.assertEqual(combined.ENC, old_encoding)

    def test_owned_label_requires_completed_adoption_evidence(self):
        r = record("complete", 2, solver="apxchol_v1")
        r["cell"].update(device="gpu", config="bg+trace_cycle/gpu-owned-q08[vec_pool_aos]")
        for value in [None, False]:
            r["metrics"]["actual_device_factor_adopted"] = value
            with self.assertRaisesRegex(ValueError, "actual device-factor adoption"):
                gpu.validate_owned_route(r)
        r["metrics"]["actual_device_factor_adopted"] = True
        gpu.validate_owned_route(r)
        for status in ["failed", "oom", "timeout", "not_converged", "n/a"]:
            r["status"] = status
            r["metrics"] = {}
            gpu.validate_owned_route(r)

    def test_snapshot_keeps_failures_metadata_without_matlab_cmg_or_bridge_substitution(self):
        import csv
        import render_snapshot
        mats = list(rc.MATRICES)
        records = []
        for index, status in enumerate(["complete", "failed", "timeout", "not_converged", "oom", "n/a"]):
            r = record(status, 2 if status == "complete" else None,
                       threads=72, solver="apxchol_v1")
            r["cell"].update(matrix_id=mats[index], device="gpu",
                config="bg+tree/gpu-owned-q02[vec_pool_aos]")
            r["provenance"] = dict(git_sha="TEST-ONLY-SOURCE", binary_sha256="TEST-ONLY-BINARY",
                sampler="gks", setup_route="owned_gpu_q02", degree_quantile=0.2,
                source_manifest_sha256="TEST-ONLY-MANIFEST", timing_protocol="TEST-ONLY")
            if status == "complete":
                r["metrics"].update(actual_device_factor_adopted=True, fp16=True,
                    factor_drop_rel=0.0001, stored_factor_nnz=12,
                    phase_observations={"setup_s": [1, 1.1, 1.2]},
                    timing_stability_warning=True)
            if status == "timeout":
                r["timeout_cap_s"] = 50
                r["matrix_meta"] = {"timeout_scope": "logical_cell"}
            records.append(r)
        bridge = record("complete", 0.001, threads=72, solver="apxchol_v1")
        bridge["cell"].update(matrix_id=mats[6], device="gpu", config="bg+tree[vec_pool_aos]")
        records.append(bridge)
        matlab_cmg = record("complete", 0.001, threads=72, solver="cmg")
        matlab_cmg["cell"].update(matrix_id=mats[0])
        records.append(matlab_cmg)
        with tempfile.TemporaryDirectory() as path, \
             mock.patch.object(chart_cells, "load_current_records", return_value=(records, {})), \
             mock.patch.object(render_snapshot.subprocess, "run") as run:
            out = pathlib.Path(path)
            (out / "figures").mkdir()
            image_names = [f"combined_{metric}{device}_{family}.png"
                           for family in ("grids", "ipm", "suitesparse")
                           for metric in ("setup", "solve", "overview")
                           for device in ("_cpu", "_gpu", "")]
            image_names += ["combined_iters_grids.png", "combined_rss_peak_ipm.png",
                            "threads_cpu_representative_setup_speedup.png",
                            "threads_gpu_solve_speedup.png"]
            image_names += [f"fill_heatmap_{family}.png" for family in ("grids", "ipm", "suitesparse")]
            for name in image_names:
                (out / "figures" / name).touch()
            render_snapshot.render(out / "TEST-ONLY-CELLS", out, 72, "TEST ONLY",
                                   sampler_comparison=True, fill_cells=out / "TEST-ONLY-FILL")
            readme = (out / "README.md").read_text()
            for name in image_names:
                self.assertIn(f"](figures/{name})", readme)
            self.assertEqual(readme.count("!["), len(image_names))
            self.assertTrue(all(line.startswith("|") for line in readme.splitlines() if "![" in line))
            coverage = json.loads((out / "coverage.json").read_text())
            with (out / "results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(coverage["expected_headline_cells"], 540)
        self.assertEqual(coverage["present"], 6)
        self.assertEqual(len(coverage["missing"]), 534)
        self.assertEqual(sum(row["solver"] == "cmg" for row in coverage["missing"]), 0)
        self.assertEqual({row["status"] for row in rows},
                         {"complete", "failed", "timeout", "not_converged", "oom", "n/a"})
        measured = next(row for row in rows if row["status"] == "complete")
        self.assertEqual(measured["actual_device_factor_adopted"], "True")
        self.assertEqual(measured["setup_route"], "owned_gpu_q02")
        self.assertEqual(measured["sampler"], "gks")
        self.assertEqual(measured["source_manifest_sha256"], "TEST-ONLY-MANIFEST")
        self.assertEqual(json.loads(measured["phase_observations"]), {"setup_s": [1, 1.1, 1.2]})
        self.assertEqual(measured["timing_stability_warning"], "True")
        self.assertEqual(run.call_count, 4)
        self.assertTrue(all("--sampler-comparison" in call.args[0] for call in run.call_args_list[:3]))
        fill_command = run.call_args_list[3].args[0]
        self.assertTrue(fill_command[1].endswith("fill_chart.py"))
        self.assertEqual(fill_command[2:], ["--cells", str(out / "TEST-ONLY-FILL"),
                                          "--out", str(out / "figures")])


if __name__ == "__main__":
    unittest.main()
