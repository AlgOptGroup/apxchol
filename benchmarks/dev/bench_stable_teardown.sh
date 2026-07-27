#!/usr/bin/env bash
# Restore CPU boost / governor / max-freq state saved by bench_stable_setup.sh.
#
# Run with:  sudo bash benchmarks/dev/bench_stable_teardown.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (sudo bash $0)" >&2
    exit 1
fi

STATE_FILE=/tmp/apxchol_bench_stable_state
BOOST_FILE=/sys/devices/system/cpu/cpufreq/boost

if [[ ! -f "$STATE_FILE" ]]; then
    echo "WARNING: $STATE_FILE missing. Restoring defaults (boost=1, governor=performance)." >&2
    [[ -f "$BOOST_FILE" ]] && echo 1 > "$BOOST_FILE"
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$g" 2>/dev/null || true
    done
    exit 0
fi

# shellcheck source=/dev/null
. "$STATE_FILE"

if [[ -f "$BOOST_FILE" && -n "${boost:-}" ]]; then
    echo "$boost" > "$BOOST_FILE"
    echo "  boost restored to $boost"
fi
if [[ -n "${governor:-}" ]]; then
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "$governor" > "$g" 2>/dev/null || true
    done
    echo "  governor restored to $governor"
fi

rm -f "$STATE_FILE"
echo "  state file removed"
