#!/usr/bin/env python3
"""Regression tests for the retained benchmark runners and scaling plots."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import runner_common as rc  # noqa: E402
import sweep_fair  # noqa: E402
import thread_scaling  # noqa: E402


def record(status, total, *, threads=16, cap=None, solver="amgcl", config=""):
    out = {
        "cell": {"family": "audit", "matrix_id": "m", "solver": solver,
                 "config": config, "threads": threads, "device": "cpu"},
        "metrics": {} if total is None else {
            "total_s": total, "stop_contract": "original-v1"},
        "status": status,
        "provenance": {"git_sha": rc.git_sha()},
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


class CompleteRouteRunnerTest(unittest.TestCase):
    def test_current_runner_forces_device_and_rejects_other_route(self):
        for device in ("cpu", "gpu"):
            output = "v1,m,2,4,1e-3,2e-3,3e-3,2,1e-9,1,750,original-v1,1,0.001," + device + "\n"
            repeats = "\n".join(f"BENCH_REPEAT phase=retained index={i} execution_route={device}"
                                for i in range(1, 4))
            for actual in (device, "gpu" if device == "cpu" else "cpu"):
                with mock.patch.object(sweep_fair, "DEVICE", device), \
                     mock.patch.object(sweep_fair, "REPS", 3), \
                     mock.patch.object(sweep_fair, "WARMUP", 0), \
                     mock.patch.object(rc, "benchmark_openmp_env", return_value={}), \
                     mock.patch.object(rc, "taskset_prefix", return_value="taskset -c 0"), \
                     mock.patch.object(sweep_fair, "sh", return_value=subprocess.CompletedProcess(
                         "benchmark", 0, output, repeats.replace("=" + device, "=" + actual))) as run:
                    status, metrics = sweep_fair.run_cpp("", "apxchol_v1", rc.APXCHOL_DEFAULT_CONFIG, False)
                self.assertIn("--v1-backend " + device, run.call_args.args[0])
                self.assertEqual(status, "complete" if actual == device else "failed")
                self.assertEqual("route_failure" in metrics, actual != device)


    def test_gpu_plan_contains_no_host_selector_ablations(self):
        self.assertEqual(sweep_fair.APX_GPU, [("apxchol_v1", sweep_fair.APX_DEFAULT_CONFIG)])


class GpuTimingIsolationTest(unittest.TestCase):
    def test_cpp_timing_does_not_start_memory_poller(self):
        output = (
            "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz\n"
            "AMGCL,m,2,4,1e-3,2e-3,3e-3,2,1e-9,1,750,original-v1,1,0.001\n"
        )
        with mock.patch.object(sweep_fair, "DEVICE", "gpu"), \
             mock.patch.object(rc, "benchmark_openmp_env", return_value={}), \
             mock.patch.object(rc, "taskset_prefix", return_value="taskset -c 0"), \
             mock.patch.object(rc, "VramSampler", side_effect=AssertionError("timing polluted")), \
             mock.patch.object(sweep_fair, "sh", return_value=subprocess.CompletedProcess(
                 "benchmark", 0, output,
                 "[bench] gpu_memory pid=9 bus_id=00000000:01:00.0 baseline_mib=238.0 "
                 "other_processes=0 other_mib=0.0\n")):
            status, metrics = sweep_fair.run_cpp("", "amgcl_cuda", "", False)
        self.assertEqual(status, "complete")
        self.assertIs(metrics["gpu_memory_polling"], False)
        self.assertNotIn("max_vram_mb", metrics)
        self.assertEqual(metrics["vram_baseline_mb"], 238.0)
        self.assertIs(metrics["gpu_shared"], False)

    def test_cpp_timeout_keeps_gpu_memory_receipt(self):
        with mock.patch.object(sweep_fair, "DEVICE", "gpu"), \
             mock.patch.object(rc, "benchmark_openmp_env", return_value={}), \
             mock.patch.object(rc, "taskset_prefix", return_value="taskset -c 0"), \
             mock.patch.object(sweep_fair, "sh", side_effect=subprocess.TimeoutExpired(
                 "benchmark", 5, output="",
                 stderr="[bench] gpu_memory pid=9 bus_id=00000000:01:00.0 baseline_mib=238.0 "
                        "other_processes=1 other_mib=512.0\n")):
            status, metrics = sweep_fair.run_cpp("", "amgcl_cuda", "", False)
        self.assertEqual(status, "timeout")
        self.assertEqual(metrics["vram_baseline_mb"], 238.0)
        self.assertIs(metrics["gpu_shared"], True)

    def test_parac_shared_calibration_and_timing_path_has_no_memory_poller(self):
        import parac_runner
        with mock.patch.object(rc, "VramSampler", side_effect=AssertionError("timing polluted")), \
             mock.patch.object(parac_runner, "sh", return_value=subprocess.CompletedProcess(
                 "gpu_rchol", 0, "APX GPU solve phase time: 0.02\nAPX stop contract: original-v1\nAPX stop checks: 1\nAPX stop check seconds: 0.001\n", "")):
            metrics = parac_runner._run_once_gpu("gpu_rchol", "matrix.mtx", 1e-8)
        self.assertEqual(metrics["solve_total"], "0.02")
        self.assertIs(metrics["gpu_memory_polling"], False)
        self.assertNotIn("vram_mb", metrics)
        self.assertNotIn("max_vram_mb", metrics)


class JuliaDriverPathTest(unittest.TestCase):
    def test_julia_driver_and_project_are_root_derived(self):
        output = (
            "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz\n"
            "AC [Kyng16;Jl],m,2,4,1e-3,2e-3,3e-3,2,1e-9,1,750,original-v1,1,0.001\n"
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


class ThreadScalingStoreTest(unittest.TestCase):
    def test_resume_checks_the_stored_identity_not_only_the_filename(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store):
            thread_scaling.emit("m", "audit", "A", "amgcl", "cfg", 4,
                                {"total_s": 1, "stop_contract": "original-v1"}, "complete")
            path = pathlib.Path(store) / "m__amgcl_cfg__t4.json"
            cell = json.loads(path.read_text())
            for field, value in (("matrix_id", "other"), ("solver", "other"),
                                 ("config", "other"), ("threads", 8), ("device", "gpu")):
                with self.subTest(field=field):
                    changed = {**cell, "cell": {**cell["cell"], field: value}}
                    path.write_text(json.dumps(changed))
                    self.assertFalse(thread_scaling.done("m", "amgcl", "cfg", 4))

    def test_render_requires_the_explicitly_selected_source_revision(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store):
            thread_scaling.emit("m", "audit", "A", "solver", "cfg", 4,
                                {"total_s": 1, "stop_contract": "original-v1"}, "complete")
            path = pathlib.Path(store) / "m__solver_cfg__t4.json"
            cell = json.loads(path.read_text())
            cell["provenance"]["git_sha"] = "b" * 40
            path.write_text(json.dumps(cell))
            self.assertFalse(thread_scaling.done("m", "solver", "cfg", 4))
            with self.assertRaisesRegex(RuntimeError, "source revision"):
                thread_scaling._scaling_records()
            with mock.patch.dict(thread_scaling.PROV, {"git_sha": "b" * 40}):
                self.assertEqual(thread_scaling._scaling_records(), [(str(path), cell)])
            cell.pop("provenance")
            path.write_text(json.dumps(cell))
            with self.assertRaisesRegex(RuntimeError, "measurement contract"):
                thread_scaling._scaling_records()

    def test_resume_rejects_malformed_records(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store):
            thread_scaling.emit("m", "audit", "A", "solver", "cfg", 4,
                                {"total_s": 1, "stop_contract": "original-v1"}, "complete")
            path = pathlib.Path(store) / "m__solver_cfg__t4.json"
            cell = json.loads(path.read_text())
            self.assertTrue(thread_scaling.done("m", "solver", "cfg", 4))
            malformed = [None, [], {}, "x", 1, True]
            for field in ("cell", "provenance", "metrics"):
                malformed.extend({**cell, field: value} for value in (None, [], ["x"], "x", 1, True))
            for value in malformed:
                with self.subTest(record=value):
                    path.write_text(json.dumps(value))
                    self.assertFalse(thread_scaling.done("m", "solver", "cfg", 4))
            path.write_text("{invalid JSON")
            self.assertFalse(thread_scaling.done("m", "solver", "cfg", 4))
            with mock.patch("builtins.open", side_effect=OSError("unreadable")):
                self.assertFalse(thread_scaling.done("m", "solver", "cfg", 4))

    def test_resume_preserves_scaling_and_device_checks(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store), \
             mock.patch.object(thread_scaling, "DEVICE", "cpu"):
            thread_scaling.emit("m", "audit", "A", "apxchol_v1", "cfg", 4,
                                {"total_s": 1, "stop_contract": "original-v1",
                                 "execution_route": "cpu"}, "complete")
            path = pathlib.Path(store) / "m__apxchol_v1_cfg__t4.json"
            cell = json.loads(path.read_text())
            self.assertTrue(thread_scaling.done("m", "apxchol_v1", "cfg", 4))
            for field in ("git_sha", "repeat", "warmup", "timing_protocol"):
                changed = {**cell, "provenance": {**cell["provenance"], field: None}}
                path.write_text(json.dumps(changed))
                self.assertFalse(thread_scaling.done("m", "apxchol_v1", "cfg", 4))
            for status in ("complete", "not_converged"):
                for total in (True, False):
                    changed = json.loads(json.dumps(cell))
                    changed["status"] = status
                    changed["cell"]["device"] = "gpu"
                    changed["metrics"]["execution_route"] = "gpu"
                    if not total:
                        changed["metrics"].pop("total_s")
                    path.write_text(json.dumps(changed))
                    self.assertFalse(thread_scaling.done("m", "apxchol_v1", "cfg", 4))

    def test_malformed_status_is_not_reused(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store):
            thread_scaling.emit("m", "audit", "A", "solver", "cfg", 4,
                                {}, "failed")
            path = pathlib.Path(store) / "m__solver_cfg__t4.json"
            cell = json.loads(path.read_text())
            for status in (None, [], {}, 0, True):
                with self.subTest(status=status):
                    cell["status"] = status
                    path.write_text(json.dumps(cell))
                    self.assertFalse(thread_scaling.done("m", "solver", "cfg", 4))
                    with self.assertRaisesRegex(RuntimeError, "measurement contract"):
                        thread_scaling._scaling_records()

    def test_malformed_identity_is_reported(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(thread_scaling, "CELLS", store):
            path = pathlib.Path(store) / "malformed.json"
            path.write_text(json.dumps({"cell": [], "status": "complete"}))
            with self.assertRaisesRegex(RuntimeError, "invalid cell identity"):
                thread_scaling._scaling_records()

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
            self.assertTrue(rc.cell_meets_contract(record))
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
                "metrics": {"total_s": 1.0, "stop_contract": "original-v1"},
                "status": "complete",
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
    def test_stale_cell_is_preserved_until_replacement_returns(self):
        old = record("complete", 3.0)
        old["provenance"]["git_sha"] = "b" * 40
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(sweep_fair, "THREADS", 16), \
             mock.patch.object(sweep_fair, "DEVICE", "cpu"), \
             mock.patch.object(sweep_fair, "build_toolchain", return_value={}), \
             mock.patch.object(rc, "benchmark_openmp_provenance", return_value={}):
            path = pathlib.Path(rc.cell_path("audit", "m", "amgcl", "", 16, "cpu"))
            path.parent.mkdir(parents=True)
            old_text = json.dumps(old)
            path.write_text(old_text)

            def failed_runner():
                self.assertEqual(path.read_text(), old_text)
                raise RuntimeError("interrupted runner")

            with self.assertRaisesRegex(RuntimeError, "interrupted runner"):
                sweep_fair.step("audit", "m", "amgcl", "", failed_runner, "AMGCL")
            self.assertEqual(path.read_text(), old_text)

            def completed_runner():
                self.assertEqual(path.read_text(), old_text)
                return "complete", {"total_s": 1.0, "stop_contract": "original-v1"}

            sweep_fair.step("audit", "m", "amgcl", "", completed_runner, "AMGCL")
            self.assertEqual(json.loads(path.read_text())["metrics"]["total_s"], 1.0)
            self.assertTrue(sweep_fair.cell_done("audit", "m", "amgcl", ""))

    def test_malformed_status_reruns_fair_sweep(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store):
            path = pathlib.Path(rc.cell_path("audit", "m", "amgcl", "", 16, "cpu"))
            path.parent.mkdir(parents=True)
            for status in (None, [], {}, 0, True):
                with self.subTest(status=status):
                    path.write_text(json.dumps(record(status, 1.0)))
                    self.assertFalse(rc.cell_done(
                        "audit", "m", "amgcl", "", 16, "cpu"))

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

    def test_timeout_without_cap_is_not_terminal_for_sweep(self):
        old = record("timeout", None)
        self.assertFalse(rc.cell_meets_contract(old))
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            path = pathlib.Path(rc.cell_path("audit", "m", "amgcl", "", 16, "cpu"))
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(old))
            with mock.patch.object(sweep_fair, "DEVICE", "cpu"):
                self.assertFalse(sweep_fair.cell_done("audit", "m", "amgcl", ""))


    def test_new_timeout_is_terminal_for_sweep(self):
        new = record("timeout", None, cap=60)
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            path = pathlib.Path(rc.cell_path("audit", "m", "amgcl", "", 16, "cpu"))
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(new))
            with mock.patch.object(sweep_fair, "DEVICE", "cpu"):
                self.assertTrue(sweep_fair.cell_done("audit", "m", "amgcl", ""))


class RelativeCapTest(unittest.TestCase):
    def test_cap_is_factor_times_best_same_device_apxchol_with_floor(self):
        self.assertEqual(sweep_fair.solve_cap([3.0, 2.0, None]), 2.0 * sweep_fair.CAP_FACTOR)
        self.assertEqual(sweep_fair.solve_cap([0.01]), sweep_fair.CAP_FLOOR_S)
        self.assertEqual(sweep_fair.solve_cap([]), float(sweep_fair.TIMEOUT))

    def test_run_over_cap_is_a_status_not_a_time(self):
        receipts = ["BENCH_REPEAT phase=warmup index=1 total_s=5.0 solver=\"x\"",
                    "BENCH_REPEAT phase=retained index=1 total_s=50.0 solver=\"x\""]
        st, m = sweep_fair.capped(("complete", {"total_s": 5.0, "setup_s": 1.0,
                                                "repeat_receipts": receipts}), 40.0)
        self.assertEqual(st, "exceeded_cap")
        self.assertNotIn("total_s", m)
        self.assertEqual(m["capped_timings"]["total_s"], 5.0)
        self.assertEqual(m["cap_s"], 40.0)

    def test_killed_run_is_exceeded_cap_and_keeps_extra(self):
        result = sweep_fair.capped(("timeout", None, {"na_reason": "killed"}), 40.0)
        self.assertEqual(result[0], "exceeded_cap")
        self.assertEqual(result[2], {"na_reason": "killed"})

    def test_run_within_cap_keeps_its_times(self):
        st, m = sweep_fair.capped(("complete", {"total_s": 3.0}), 40.0)
        self.assertEqual((st, m["total_s"], m["cap_s"]), ("complete", 3.0, 40.0))

    def test_exceeded_cap_cells_record_their_cap_and_are_terminal(self):
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            with self.assertRaises(ValueError):
                rc.emit_cell("audit", "m", "amgcl", "", "exceeded_cap", {}, 16, "cpu",
                             {"git_sha": rc.git_sha()})
            rc.emit_cell("audit", "m", "amgcl", "", "exceeded_cap", {"cap_s": 40.0}, 16, "cpu",
                         {"git_sha": rc.git_sha()}, timeout_cap_s=40.0)
            self.assertTrue(rc.cell_done("audit", "m", "amgcl", "", 16, "cpu"))
            self.assertFalse(rc.cell_done("audit", "m", "amgcl", "", 16, "cpu", rerun_timeouts=True))


class SourceIdentityTest(unittest.TestCase):
    SHA = "1c4abb35f1cf08276bc9dc65534dbfb6d2b66059"

    def setUp(self):
        rc.git_sha.cache_clear()            # git_sha is cached per process
        self.addCleanup(rc.git_sha.cache_clear)

    def head(self, out):
        return mock.patch.object(rc, "sh", return_value=mock.Mock(stdout=out))

    def test_tarball_source_uses_declared_commit(self):
        with self.head(""), mock.patch.dict(os.environ, {"APXCHOL_SOURCE_COMMIT": self.SHA}):
            self.assertEqual(rc.git_sha(), self.SHA)

    def test_declared_commit_must_match_checkout_and_be_full(self):
        with self.head("0" * 40 + "\n"), mock.patch.dict(os.environ, {"APXCHOL_SOURCE_COMMIT": self.SHA}):
            with self.assertRaises(ValueError):
                rc.git_sha()
        rc.git_sha.cache_clear()
        with self.head(""), mock.patch.dict(os.environ, {"APXCHOL_SOURCE_COMMIT": "1c4abb35"}):
            with self.assertRaises(ValueError):
                rc.git_sha()


class FairSweepSelectionTest(unittest.TestCase):
    def test_full_fair_plan_denominator_excludes_retired_storage(self):
        selected = list(sweep_fair.selected_matrices(
            {"grids", "suitesparse", "ipm"}))
        self.assertEqual(sweep_fair.APX_GPU[0],
                         ("apxchol_v1", sweep_fair.APX_DEFAULT_CONFIG))
        self.assertEqual(len(sweep_fair.APX_GPU), len(set(sweep_fair.APX_GPU)))
        self.assertTrue(all(retired not in config
                            for _, config in sweep_fair.APX + sweep_fair.APX_GPU
                            for retired in ("[vec]", "[bstr]", "[vec_pool]", "[fwd_star]", "[forward_star]")))
        self.assertEqual(sweep_fair.planned_cell_count(selected, "cpu"), 351)
        self.assertEqual(sweep_fair.planned_cell_count(selected, "gpu"), 162)
        self.assertEqual(
            sweep_fair.planned_cell_count(selected, "cpu")
            + sweep_fair.planned_cell_count(selected, "gpu"),
            513,
        )

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


class CmgAvailabilityTest(unittest.TestCase):
    def test_legacy_adapter_stops_before_export_or_solver_execution(self):
        cmg = sweep_fair.cmg_matlab_runner
        with mock.patch.object(cmg, "_dump") as export, \
             mock.patch.object(cmg, "sh") as run, \
             mock.patch.object(rc, "emit_cell") as emit:
            ready, reason = cmg.available()
            self.assertFalse(ready)
            self.assertIn("local RHS", reason)
            self.assertEqual(cmg.run_one("grid_500"), "UNAVAILABLE: " + reason)
        export.assert_not_called()
        run.assert_not_called()
        emit.assert_not_called()

    def test_sweep_reports_unavailable_cmg_once_before_first_matrix(self):
        state = {"probed": False, "ok": False, "why": ""}
        def matrix(mid, *_args):
            self.assertTrue(state["probed"])
            sweep_fair.run_cmg(mid)
            sweep_fair.run_cmg(mid)
        argv = ["sweep_fair.py", "--only", "grid_500", "--no-julia", "--no-parac"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(sweep_fair, "_CMG_STATE", state), \
             mock.patch.object(sweep_fair, "RUN_CMG", True), \
             mock.patch.object(sweep_fair, "JULIA", []), \
             mock.patch.object(sweep_fair, "RUN_PARAC", False), \
             mock.patch.object(sweep_fair, "do_matrix", side_effect=matrix), \
             mock.patch.object(sweep_fair.cmg_matlab_runner, "run_one") as run, \
             mock.patch("builtins.print") as output:
            sweep_fair.main()
        unavailable = [call for call in output.call_args_list
                       if "UNAVAILABLE" in str(call.args[0])]
        self.assertEqual(len(unavailable), 1)
        run.assert_not_called()


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


if __name__ == "__main__":
    unittest.main()
