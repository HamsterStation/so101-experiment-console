#!/bin/bash
set -euo pipefail
case "$1" in student.onnx|student-w8a8.onnx) ;; *) exit 1;; esac
name=$1
gpu_dir=robotlab/outputs/act-cube46-comparison-v1
edge_dir=cie6007-edge/act-comparison/cube46-v1
sum=$(ssh gpu4080 "cd $gpu_dir && sha256sum $name" | cut -d' ' -f1)
[[ "$sum" =~ ^[a-f0-9]{64}$ ]]
ssh gpu4080 "cat $gpu_dir/$name" | ssh jetson-lan "cd $edge_dir && cat > $name.part && echo '$sum  $name.part' | sha256sum -c - && mv $name.part $name"
