# 240 epochs ddim 
#paired
    folder1="/home/nervld/gitclone/diffusers/data/vton_test_result/1"
    folder2="/home/nervld/gitclone/diffusers/data/vton_test/real_images"
    python ./PerceptualSimilarity/lpips_2dirs.py -d0 $folder1 -d1 $folder2 --use_gpu
    python ./test_SSIM.py --folder1 $folder1 --folder2 $folder2
    python -m pytorch_fid $folder1 $folder2 --device cuda:0

# 0.07231199047248628   0.896280889575317   8.532824851216049

