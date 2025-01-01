bsmult=$1 # Was 8
for scale in 4 8 16 2 1; do size=$(( 64 / scale)); stride=$(( scale / 4 )); \
  for bs in $(( 32768 / $bsmult / $size / $size )); do \
    for complexity in 64; do \
        ./killall.sh; python 1x1orig.py \
        --name V100_s1_336-901v_c$complexity --batch_size $bs --val_batch_size $bs --seed 0 --lr 0.0005 \
        --scale $scale --size $size --tile_size $size --stride $stride --val_size $size --val_stride $(( stride * 2 )) \
        --epochs $(( scale * 8 )) --model pygo1x1 --complexity $complexity; done; done; done
