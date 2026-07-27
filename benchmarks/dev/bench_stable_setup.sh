#!/usr/bin/env bash
# Configure the system for thermal-stable benchmarking. Requires sudo.
#
# What it does:
#   1. Disables CPU boost (locks the upper frequency at the base clock).
#      Without this, cores swing between base and turbo depending on thermal +
#      power state + sibling-core load → up to 25-30% session variance.
#   2. Leaves governor at "performance" — that just pins to scaling_max_freq.
#   3. Saves prior state so bench_stable_teardown.sh can restore it.
#
# After this runs, `cat /sys/devices/system/cpu/cpufreq/boost` reports 0
# and all cores' scaling_cur_freq should sit at the base frequency.
#
# Run with:  sudo bash benchmarks/dev/bench_stable_setup.sh
# Restore:   sudo bash benchmarks/dev/bench_stable_teardown.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (sudo bash $0)" >&2
    exit 1
fi

STATE_FILE=/tmp/apxchol_bench_stable_state
BOOST_FILE=/sys/devices/system/cpu/cpufreq/boost

if [[ ! -f "$BOOST_FILE" ]]; then
    echo "WARNING: $BOOST_FILE missing (non-AMD or stripped kernel?)" >&2
fi

# Snapshot prior state for teardown.
{
    [[ -f "$BOOST_FILE" ]] && echo "boost=$(cat "$BOOST_FILE")"
    echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "scaling_max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
} > "$STATE_FILE"

# Disable boost.
if [[ -f "$BOOST_FILE" ]]; then
    echo 0 > "$BOOST_FILE"
    echo "  boost disabled (was 1)"
fi

# Force performance governor on all cores (just in case).
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g" 2>/dev/null || true
done

# Verify.
echo ""
echo "=== Post-setup state ==="
echo "boost: $(cat "$BOOST_FILE" 2>/dev/null || echo n/a)"
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "scaling_max_freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) kHz"
echo ""
echo "Saved prior state to $STATE_FILE."
echo "Restore with: sudo bash benchmarks/dev/bench_stable_teardown.sh"
