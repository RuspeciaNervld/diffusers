# 240 epochs ddim 
#paired
    folder1="/home/nervld/gitclone/diffusers/output/vton_test_small_2/catvton"
    folder2="/home/nervld/gitclone/diffusers/data/vton_test_small_2/real_images"
    python /home/nervld/gitclone/diffusers/calculate_metrics/PerceptualSimilarity/lpips_2dirs.py -d0 $folder1 -d1 $folder2 --use_gpu | awk -F: '{sum+=$2} END {print "Mean LPIPS value: " sum/NR}'
    python /home/nervld/gitclone/diffusers/calculate_metrics/test_SSIM.py --folder1 $folder1 --folder2 $folder2
    # python -m pytorch_fid $folder1 $folder2 --device cpu
    python -m pytorch_fid $folder1 $folder2 --device cuda:0

# 0.07231199047248628   0.896280889575317   8.532824851216049

#100 test
# Name                                 lpips(↓)         SSIM(↑)                  FID(↓)
#inference_results_warp_2000          0.173176   0.814014414061769        48.57734535694672
#lpips6000_dream50_batch2_lr1e-6      0.133      0.8226443841010869       39.81517248161762
#nolpips6000_dream50_batch2_lr1e-5    0.168814      0.8098030181406536    46.05753043984501
#catvton_origin_warp                  0.119127     0.8285121906267594     37.82335712662723
#catvton_no_warp_no_detail *           0.105059     0.8390654698112263    31.57171982202553
#catvton_no_warp_with_detail          0.111461     0.8380975008211948     32.374856566683775
#像素四宫格_独立学习_step_1500          0.138765    0.8113227504985672      38.656401777310634
#像素四宫格_同步学习_step_1000          0.126373    0.8211365343472155      35.636567547479785
#scale_dream10_无人warp_step_500       0.111912   0.8547782200089791       30.10919866182934
#scale_dream10_无人warp_step_1000     0.109265     0.8579005143264417      30.36955137394787
#scale_dream10_无人warp_step_1500     0.112373    0.8585452456374005      31.68971133351164  
#                                   0.105161      0.8557868991662659        42.09912547188608
#catvton                            0.0969355      0.842252172101854        37.48109394733717

#100 test update
# Name                                  lpips         SSIM                  FID
#catvton_no_warp_no_detail *           0.0948431   0.8735463802804877   28.167530807231827
#像素四宫格_独立学习_step_1500          0.122196    0.8483158596611524    34.76572992020695
#像素四宫格_同步学习_step_1000          0.111078    0.8571941127942867    32.46887523924019
#