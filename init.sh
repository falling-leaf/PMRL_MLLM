python3 start_code.py --device 0 --sub_device 0 --method wise --model blip2 --ds caption --pmrl_tau_alignment 8 --pmrl_scale 1 --num_rephrase 4 --using_extra
rm -rf /root/PMRL_MLLM/tmp_rephrase_samples
mv /root/PMRL_MLLM/saved_rephrase_samples /root/PMRL_MLLM/tmp_rephrase_samples
python3 start_code.py --device 0 --sub_device 0 --method wise --model blip2 --ds caption
mv /root/PMRL_MLLM/saved_rephrase_samples /root/PMRL_MLLM/tmp_base_samples
python3 visualization.py