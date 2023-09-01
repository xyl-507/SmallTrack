export PYTHONPATH=/home/xyl/xyl-code/SmallTrack:$PYTHONPATH

START=11
END=20
for i in $(seq $START $END)
do
    python -u ../../tools/test.py \
        --snapshot snapshot/checkpoint_e$i.pth \
	      --config config-smalltrack.yaml \
	      --gpu_id $(($i % 2)) \
	      --dataset DTB70 2>&1 | tee logs/test_dataset.log &
done
# python ../../tools/eval.py --tracker_path ./results --dataset UAV123 --num 4 --tracker_prefix 'ch*'

# learn from siammask
