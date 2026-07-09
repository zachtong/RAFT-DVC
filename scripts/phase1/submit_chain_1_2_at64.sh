#!/bin/bash
# =============================================================================
# One-shot UNATTENDED submission of the 1/2@64 run across the 24 h walltime cap.
# =============================================================================
# The `gh` partition kills a job at 24 h.  This submits a CHAIN of N jobs: each
# one waits (afterany = regardless of how the previous ended, including a
# walltime kill) for its predecessor, then auto-resumes from latest.pth.  So you
# run this ONCE and the training continues on its own for up to N*24 h.
#
# The run needs ~60 h (about 14 min/epoch over the remaining ~240 epochs), so the
# default N=4 (96 h) covers it with slack.  Once training reaches epoch 300, the
# completion guard in slurm_1_2_at64_vista.sh makes trailing jobs exit in seconds.
#
# IMPORTANT: only ONE job for this experiment may run at a time (two would both
# write latest.pth and corrupt it).  Before running this, cancel any existing
# rdvc12at64 job:  squeue -u $USER   then   scancel <jobid>.
#
# Usage (from $WORK/projects/RAFT-DVC):
#     bash scripts/phase1/submit_chain_1_2_at64.sh [N]     # N defaults to 4
# =============================================================================
set -euo pipefail

N=${1:-4}
SCRIPT=scripts/phase1/slurm_1_2_at64_vista.sh

cd "$WORK/projects/RAFT-DVC"
mkdir -p logs

jid=$(sbatch --parsable "$SCRIPT")
echo "window 1: job $jid  (queued now)"
for i in $(seq 2 "$N"); do
    jid=$(sbatch --parsable --dependency=afterany:"$jid" "$SCRIPT")
    echo "window $i: job $jid  (starts after the previous window ends)"
done

echo
echo "Chain of $N submitted -- unattended for up to $((N * 24)) h."
echo "Each window resumes from latest.pth; trailing jobs no-op once at epoch 300."
echo "Monitor:  squeue -u \$USER"
echo "Live log: tail -f logs/rdvc_1_2_at64_<jobid>.out   (of the RUNNING window)"
