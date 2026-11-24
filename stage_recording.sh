#!/bin/bash
# Run on the Mac: pass the ABSOLUTE directory of a closed Jetson v2.1 recording.
# This copies a completed recording, retains the source and original copy, and does not upload to the Hub.
set -eu -o pipefail
src=${1:?Provide the absolute Jetson recording directory}; case "$src" in /home/nvidia/*) ;; *) echo 'Expected an absolute recording path below /home/nvidia' >&2; exit 2;; esac
[[ "$src" =~ ^/[a-zA-Z0-9_./-]+$ && "$src" != *..* ]] || { echo 'Unsupported source path' >&2; exit 2; }
ssh -o BatchMode=yes jetson "test -s '$src/meta/info.json' && test -d '$src/data' && test -d '$src/videos'" || { echo 'Recording files not found' >&2; exit 2; }
run=$(ssh -o BatchMode=yes gpu4080 'mkdir -p ~/robotlab/cie6007-recordings && mktemp -d ~/robotlab/cie6007-recordings/run-XXXXXX'); scratch=$(mktemp -d); trap 'rm -rf -- "$scratch"' EXIT
rsync -a -e 'ssh -o BatchMode=yes' --include='/meta/***' --include='/data/***' --include='/videos/***' --exclude='*' "jetson:$src/" "$scratch/"
rsync -a -e 'ssh -o BatchMode=yes' "$scratch/" "gpu4080:$run/dataset/"
scp -q "$(dirname "$0")/verify_recording.py" "gpu4080:$run/"
ssh -o BatchMode=yes gpu4080 "cd '$run' && export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 && ~/robotlab/.venv/bin/python verify_recording.py before && ~/robotlab/.venv/bin/python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=local/so101_recording --root='$run/dataset' --push-to-hub=false --force-conversion && ~/robotlab/.venv/bin/python verify_recording.py after"
echo "4080 dataset: $run/dataset"
