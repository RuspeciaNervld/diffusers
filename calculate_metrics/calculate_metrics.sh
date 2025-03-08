# 240 epochs ddim 
#paired
    folder1="/home/nervld/gitclone/diffusers/output/vton_test_small/inference_results_catvton_small"
    folder2="/home/nervld/gitclone/diffusers/data/vton_test_small/real_images"
    python /home/nervld/gitclone/diffusers/calculate_metrics/PerceptualSimilarity/lpips_2dirs.py -d0 $folder1 -d1 $folder2 --use_gpu | awk -F: '{sum+=$2} END {print "Mean LPIPS value: " sum/NR}'
    python /home/nervld/gitclone/diffusers/calculate_metrics/test_SSIM.py --folder1 $folder1 --folder2 $folder2
    python -m pytorch_fid $folder1 $folder2 --device cuda:0

# 0.07231199047248628   0.896280889575317   8.532824851216049

#100 test
#inference_results_warp_2000   0.173176   0.814014414061769   48.57734535694672

