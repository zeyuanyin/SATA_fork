CUDA_VISIBLE_DEVICES=7 \
python main-eval.py --model_name "deit_base_patch16_224" --gamma 0.7 --data_path /research/cvlshare/cvl-zeyuan/imagenet/val --sata --batch_size 256