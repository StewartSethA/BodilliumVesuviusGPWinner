#bsmult=$1 # Was 8
scale=$1
size=$2
stride=$(( size / 2 ))
out_size=$3
complexity=$4
bsmult=$5
model=pygo1x1 #i3d #pygo1x1 #i3d
epochsmult=16
#for scale in 4 8 16 2 1; do size=$(( 64 / scale)); stride=$(( scale / 4 )); \

#for scale in $3; do size=$(( 64 / scale)); stride=$(( scale / 4 )); \
  #for bs in $(( 32768 / $bsmult / $size / $size )); do \
  #for bs in 1; do \
for scale in 8 1 2 4; do
for size in 16 32 64; do # 64; do
out_size=$size
stride=$(( size / 2 ))
complexity=8
bsmult=64
epochsmult=4
bs=$bsmult;
    for complexity in 4 8 16 24 32 64; do \
        bs=1 #$(( 32 / complexity )) #$(( 512 * 4 / complexity )) #64 * 4096 * 64 / complexity / complexity / size / size ))
        ./killall.sh
        echo python 1x1orig.py \
        --name Dec13_2040REALDELTA001_s1_336-901v_c$complexity --batch_size $bs --val_batch_size $bs --seed 0 --lr 0.00002 \
        --scale $scale --size $size --tile_size $size --stride $stride --val_size $size --val_stride $(( stride  )) \
        --epochs $(( scale * $epochsmult )) --model $model --complexity $complexity --out_size $out_size;
        python 1x1orig.py \
        --name Dec13_2040REALDELTA001_s1_336-901v_c$complexity --batch_size $bs --val_batch_size $bs --seed 0 --lr 0.00002 \
        --scale $scale --size $size --tile_size $size --stride $stride --val_size $size --val_stride $(( stride  )) \
        --epochs $(( scale * $epochsmult )) --model $model --complexity $complexity --out_size $out_size; done #; done

done
done
