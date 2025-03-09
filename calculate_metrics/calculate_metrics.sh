# 240 epochs ddim 
#paired
    folder1="/home/nervld/gitclone/diffusers/output/vton_test_small/catvton_no_warp_with_detail"
    folder2="/home/nervld/gitclone/diffusers/data/vton_test_small/real_images"
    python /home/nervld/gitclone/diffusers/calculate_metrics/PerceptualSimilarity/lpips_2dirs.py -d0 $folder1 -d1 $folder2 --use_gpu | awk -F: '{sum+=$2} END {print "Mean LPIPS value: " sum/NR}'
    python /home/nervld/gitclone/diffusers/calculate_metrics/test_SSIM.py --folder1 $folder1 --folder2 $folder2
    python -m pytorch_fid $folder1 $folder2 --device cuda:0

# 0.07231199047248628   0.896280889575317   8.532824851216049

#100 test
# Name                                  lpips         SSIM                  FID
#inference_results_warp_2000          0.173176   0.814014414061769   48.57734535694672
#lpips6000_dream50_batch2_lr1e-6      0.133      0.8226443841010869  39.81517248161762
#nolpips6000_dream50_batch2_lr1e-5    0.168814      0.8098030181406536  46.05753043984501
#catvton_origin_warp                  0.119127     0.8285121906267594   37.82335712662723
#catvton_no_warp_no_detail            0.105059     0.8390654698112263    31.57171982202553
#catvton_no_warp_with_detail          0.111461     0.8380975008211948    32.374856566683775

#whole test
# Name                                  lpips         SSIM                  FID
#inference_results_warp_2000          0.173176   0.814014414061769   48.57734535694672
#lpips6000_dream50_batch2_lr1e-6      0.133      0.8226443841010869  39.81517248161762
#nolpips6000_dream50_batch2_lr1e-5    0.168814      0.8098030181406536  46.05753043984501
#catvton_origin_warp                  0.119127     0.8285121906267594   37.82335712662723
