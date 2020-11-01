# Dynapro
Command to run
CUDA_VISIBLE_DEVICES=1 python examples/run_propara.py --model_type dynapro --model_name_or_path bert-base-uncased  --do_lower_case   --train_file ../data/train-gl-lc-one-v1.1.json --predict_file ../data/dev-gl-lc-one-v1.1.json   --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8  --learning_rate 3e-5  --num_train_epochs 100.0   --max_seq_length 200   --doc_stride 128   --output_dir [Path to save output models] --[do_train/do_eval] [--overwrite_output_dir] [--overwrite_cache] 
