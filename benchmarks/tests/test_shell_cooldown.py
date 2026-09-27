"""Exercise the real cooldown functions without sensors, sleeping, or solvers."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

DEV = Path(__file__).resolve().parents[1] / "dev"
HELPERS = {
    "stable": ("bench_stable.sh", "# ── Main", "wait_cooldown 70"),
    "reference": ("bench_reference_table.sh", "WORKLOADS=", "wait_cool 70"),
}
SCENARIOS = ("missing", "failure", "empty", "cool", "hot_then_cool")


class ShellCooldownTest(unittest.TestCase):
    def check_cooldown(self, helper, scenario, inherit_errexit):
        bash = shutil.which("bash")
        awk = shutil.which("awk")
        dirname = shutil.which("dirname")
        if not all((bash, awk, dirname)):
            self.skipTest("bash, awk and dirname required")
        filename, marker, call = HELPERS[helper]
        source = (DEV / filename).read_text()
        self.assertEqual(source.count(marker), 1)
        # Keep the original function definitions and strict shell options;
        # omit only the workload loop. No solver command is reached.
        prefix = source.split(marker, 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / "bin"
            tools.mkdir()
            for name, target in (("awk", awk), ("dirname", dirname)):
                (tools / name).symlink_to(target)
            (root / "openmp_affinity.sh").write_bytes((DEV / "openmp_affinity.sh").read_bytes())
            sleeps = root / "sleeps"
            sleep = tools / "sleep"
            sleep.write_text("#!" + bash + '\nprintf "sleep\\n" >> "$SLEEP_LOG"\n')
            sleep.chmod(0o700)
            if scenario != "missing":
                sensor = tools / "sensors"
                bodies = {
                    "failure": "exit 1\n",
                    "empty": "exit 0\n",
                    "cool": "printf 'Tctl: +63.4°C\\n'\n",
                    "hot_then_cool": "if [[ -e \"$SLEEP_LOG\" ]]; then printf 'Tctl: +63.4°C\\n'; else printf 'Tctl: +78.0°C\\n'; fi\n",
                }
                sensor.write_text("#!" + bash + "\n" + bodies[scenario])
                sensor.chmod(0o700)
            fixture = root / "fixture.sh"
            fixture.write_text(prefix + "\n" + call + "\nprintf 'COOLDOWN_RETURNED\\n'\n")
            env = dict(os.environ, PATH=str(tools), QUIET="1", SLEEP_LOG=str(sleeps))
            for key in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"):
                env.pop(key, None)
            command = [bash, "--noprofile", "--norc", "-O" if inherit_errexit else "+O", "inherit_errexit", str(fixture), "grid_2"]
            completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=5)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "COOLDOWN_RETURNED\n")
            self.assertEqual(sleeps.read_text() if sleeps.exists() else "", "sleep\n" if scenario == "hot_then_cool" else "")


def _case(helper, scenario, inherit_errexit):
    def test(self):
        self.check_cooldown(helper, scenario, inherit_errexit)
    return test


for _helper in HELPERS:
    for _scenario in SCENARIOS:
        for _inherit in (False, True):
            setattr(ShellCooldownTest, "test_{}_{}_inherit_{}".format(_helper, _scenario, int(_inherit)), _case(_helper, _scenario, _inherit))


if __name__ == "__main__":
    unittest.main()
