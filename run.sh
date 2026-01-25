#!/bin/bash
# Array of values for pmrl_tau_alignment and pmrl_scale
# alignment_values=(0.05 0.1 0.5 1)
# scale_values=(0.05 0.1 0.5 1)
# alignment_values=(0.05 0.1 0.5 1 2 4 8)
# scale_values=(0.05 0.1 0.5 1 2 4 8)
# python3 start_code.py --device 0 --sub_device 0 --method wise --model blip2 --ds caption --pmrl_tau_alignment 1 --pmrl_scale 1 --num_rephrase 1
scale_values=(1 5 10 15)
alignment_values=(1 2 4 8)
num_rephrase_values=(4)
# alignment_values=(1 2 4 8)
# scale_values=(10 15 20 25)
# num_rephrase_values=(3 4 5)

# Loop through all combinations of alignment and scale
for alignment in "${alignment_values[@]}"; do
    for scale in "${scale_values[@]}"; do
        for num_rephrase in "${num_rephrase_values[@]}"; do
            echo "Running with pmrl_tau_alignment=$alignment, pmrl_scale=$scale, num_rephrase=$num_rephrase"
            python3 start_code.py --device 0 --sub_device 0 --method wise --model llava --ds vqa --pmrl_tau_alignment $alignment --pmrl_scale $scale --num_rephrase $num_rephrase
            echo "Completed run with pmrl_tau_alignment=$alignment, pmrl_scale=$scale, num_rephrase=$num_rephrase"
            echo "----------------------------------------"
        done
    done
done

echo "All combinations completed!"