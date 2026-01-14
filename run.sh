#!/bin/bash
# Array of values for pmrl_tau_alignment and pmrl_scale
alignment_values=(0.05 0.1 0.5 1)
scale_values=(0.05 0.1 0.5 1)

# Loop through all combinations of alignment and scale
for alignment in "${alignment_values[@]}"; do
    for scale in "${scale_values[@]}"; do
        echo "Running with pmrl_tau_alignment=$alignment and pmrl_scale=$scale"
        python3 start_code.py --device 0 --sub_device 0 --method wise --model blip2 --ds caption --pmrl_tau_alignment $alignment --pmrl_scale $scale
        echo "Completed run with pmrl_tau_alignment=$alignment and pmrl_scale=$scale"
        echo "----------------------------------------"
    done
done

echo "All combinations completed!"