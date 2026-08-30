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
import combined_charts as combined  # noqa: E402
import gpu_charts as gpu  # noqa: E402
import runner_common as rc  # noqa: E402
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


class ThreadScalingStoreTest(unittest.TestCase):
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
            self.assertTrue(thread_scaling.done("m", "solver", "cfg", 4))

            record.pop("schema")
            path.write_text(json.dumps(record))
            self.assertFalse(thread_scaling.done("m", "solver", "cfg", 4))
            with self.assertRaises(RuntimeError):
                thread_scaling._scaling_records()

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
                        threads=16, device="gpu", prov={})
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
                         {"total_s": 7.2}, 72, "gpu", {})
            self.assertEqual(sweep_fair.gpu_apx_total("audit", "m"), 7.2)


class FairSweepSelectionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
