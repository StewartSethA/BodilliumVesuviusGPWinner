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
for scale in 8 16 32 4 2 1; do
for size in 64 32 16; do
out_size=$size
stride=$(( size / 2 ))
complexity=32
bsmult=6
bs=$bsmult;
    for complexity in $complexity; do \
        python 1x1orig.py \
        --name Nov18_s1_336-901v_c$complexity --batch_size $bs --val_batch_size $bs --seed 0 --lr 0.00002 \
        --scale $scale --size $size --tile_size $size --stride $stride --val_size $size --val_stride $(( stride  )) \
        --epochs $(( scale * $epochsmult )) --model $model --complexity $complexity --out_size $out_size; done #; done

done
done