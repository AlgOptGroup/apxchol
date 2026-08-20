#!/usr/bin/env bash
# CSCS Daint usage at a glance, without the web portal.
#
#   scripts/cscs_usage.sh            # this month
#   scripts/cscs_usage.sh 2026-08-19 # since a date
#
# Node-hours are what CSCS bills: a job is charged NNodes x Elapsed even if it
# uses one of the node's four GH200 modules, so `sacct` elapsed alone understates
# nothing here but per-module parallelism is free -- run 4 configs per node.
set -u
SINCE="${1:-$(date +%Y-%m-01)}"
ACCOUNT="${APXCHOL_CSCS_ACCOUNT:-prep34}"
HOST="${APXCHOL_CSCS_HOST:-daint}"

ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" bash -s -- "$SINCE" "$ACCOUNT" <<'REMOTE' 2>/dev/null
SINCE="$1"; ACCOUNT="$2"
printf '=== CSCS %s usage since %s ===\n\n' "$ACCOUNT" "$SINCE"

printf 'project total, by user (node-hours):\n'
sreport -n -P cluster AccountUtilizationByUser start="$SINCE" -t hours --tres=node account="$ACCOUNT" 2>/dev/null |
  awk -F'|' '{u=($3==""?"(project total)":$3); printf "  %-14s %8s h  %s\n", u, $6, $4}'

printf '\nyour jobs (fine-grained, node-seconds x nodes):\n'
sacct -X -n -P -u "$USER" -S "$SINCE" --format=JobID,Elapsed,NNodes,State,Partition 2>/dev/null |
  awk -F'|' '
    { split($2,t,":"); s=t[1]*3600+t[2]*60+t[3]; ns=s*$3; tot+=ns; n++;
      st[$4]++; if (ns>max) { max=ns; maxj=$1 } }
    END {
      if (n==0) { print "  (none)"; exit }
      printf "  %d jobs, %.0f node-seconds = %.3f node-hours\n", n, tot, tot/3600
      printf "  longest: job %s at %.0f node-seconds\n", maxj, max
      printf "  states:"; for (k in st) printf " %s=%d", k, st[k]; printf "\n"
    }'

printf '\nrunning / queued now:\n'
q=$(squeue -h -u "$USER" 2>/dev/null); [ -n "$q" ] && echo "$q" | sed 's/^/  /' || echo "  (nothing)"

printf '\nscratch:\n'
lfs quota -h -u "$USER" /capstor/scratch/cscs/"$USER" 2>/dev/null | sed -n '2,3p' | sed 's/^/  /' ||
  du -sh /capstor/scratch/cscs/"$USER" 2>/dev/null | sed 's/^/  /'
REMOTE
