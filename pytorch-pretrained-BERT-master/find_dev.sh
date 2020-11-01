#!/bin/bash
#on tmux 0
# CUDA_VISIBLE_DEVICES=0 python3.5 examples/global_local_with_decoder.py --bert_model bert-base-uncased --do_lower_case   --train_file ../data/train-gl-lc-one-v1.1.json   --predict_file ../data/dev-gl-lc-one-v1.1.json   --train_batch_size 8 --predict_batch_size 8  --learning_rate 3e-5   --num_train_epochs 50.0   --max_seq_length 200   --doc_stride 128   --output_dir /slowdrive/dec1/original_before_one_full_contextspan_class_action_decoder_no_lstm  --do_train 
# CUDA_VISIBLE_DEVICES=1 python3.5 examples/run_propara.py --model_type span_class_action_decoder_no_lstm --model_name_or_path bert-base-uncased  --do_lower_case   --train_file ../data/train-gl-lc-one-v1.1.json --predict_file ../data/dev-gl-lc-one-v1.1.json   --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8  --learning_rate 3e-5  --num_train_epochs 50.0   --max_seq_length 200   --doc_stride 128   --output_dir /slowdrive/dec1/original_before_one_full_contextspan_class_action_decoder_no_lstm --do_train
best_i="0"
best_i_f1="0.00"
# for i in `seq 10 100`
for i in `seq 6 49`
do
    # CUDA_VISIBLE_DEVICES=0 python3.5 examples/run_propara.py --model_type original_before_with_cls --model_name_or_path bert-base-uncased  --do_lower_case   --train_file ../data/train-gl-lc-one-v1.1.json --predict_file ../data/dev-gl-lc-one-v1.1.json   --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8  --learning_rate 3e-5  --num_train_epochs 50.0   --max_seq_length 200   --doc_stride 128   --output_dir /slowdrive/jan25/ablation_no_lstm_in_class_prediction_dropout_01/ --ablation original_before_with_cls  --do_eval
    cd ../data
    cp  /slowdrive/apr8/ablate_basic_bert_action_beftore_one/predictions_${i}_ablate_basic_bert_action_beftore_.json .
    cp  /slowdrive/apr8/ablate_basic_bert_action_beftore_one/b_predictions_${i}_ablate_basic_bert_action_beftore_.json .
    cp  /slowdrive/apr8/ablate_basic_bert_action_beftore_one/action_predictions_${i}_ablate_basic_bert_action_beftore_.json . # python2 make_evaluaion_files.py --data_devision  dev --use_actions >o.txt
    python2 make_evaluaion_files.py --data_devision  dev --use_actions --prediction_file predictions_${i}_ablate_basic_bert_action_beftore_.json --before_state_prediction_file  b_predictions_${i}_ablate_basic_bert_action_beftore_.json --action_prediction_file action_predictions_${i}_ablate_basic_bert_action_beftore_.json > o.txt
# python2 make_evaluaion_files.py --data_devision  dev --use_actions --prediction_file predictions_98_ablation_no_lstm_in_class_prediction_.json --before_state_prediction_file  b_predictions_98_ablation_no_lstm_in_class_prediction_.json --action_prediction_file action_predictions_98_ablation_no_lstm_in_class_prediction_.json
# python2 make_evaluaion_files.py --data_devision  dev --use_actions --prediction_file predictions_41_ablation_no_lstm_in_span_prediction_.json --before_state_prediction_file  b_predictions_41_ablation_no_lstm_in_span_prediction_.json --action_prediction_file action_predictions_41_ablation_no_lstm_in_span_prediction_.json
# python2 make_evaluaion_files.py --data_devision  dev --use_actions --prediction_file predictions_49_all_current_predictions_.json --before_state_prediction_file  b_predictions_49_ablation_no_lstm_in_class_prediction_.json --action_prediction_file action_predictions_49_all_current_predictions_.json>o.txt
# python2 make_evaluaion_files.py --data_devision  dev --use_actions --prediction_file predictions_23_ablation_no_lstm_in_class_prediction_.json --before_state_prediction_file  b_predictions_23_ablation_no_lstm_in_class_prediction_.json --action_prediction_file action_predictions_23_ablation_no_lstm_in_class_prediction_.json
    # python2 make_evaluaion_files.py --data_devision  dev --use_actions --prediction_file predictions_6_ablation_no_lstm_in_class_prediction_.json --before_state_prediction_file  b_predictions_6_ablation_no_lstm_in_class_prediction_.json --action_prediction_file action_predictions_6_ablation_no_lstm_in_class_prediction_.json>o.txt
    
    # python2 make_evaluaion_files.py --data_devision  dev --prediction_file predictions_${i}_ablation_no_lstm_in_class_prediction_.json --before_state_prediction_file  b_predictions_${i}_ablation_no_lstm_in_class_prediction_.json >o.txt
    # python2 make_evaluaion_files.py --data_devision  dev --use_actions --prediction_file predictions_20_ablation_no_lstm_in_class_prediction_.json --before_state_prediction_file  b_predictions_20_ablation_no_lstm_in_class_prediction_.json --action_prediction_file action_predictions_20_ablation_no_lstm_in_class_prediction_.json>o.txt
  
    cd ../propara_eval
    python evaluator/evaluator.py --predictions data/test/propara_format_predictions_action.tsv --answers data/test/answers.tsv >out.txt
    python evaluator/evaluator.py --predictions data/dev/propara_format_predictions_action.tsv --answers data/dev/answers.tsv >out.txt
    # python naacl_eval_refinement.py
    # python3.6 evalQA.py para_id.dev.txt refined_all.chain.dev.procAll.v3.gold naacl_format_preds.tsv
    # python3.6 evalQA.py para_id.test.txt all.chain.test.original.v3.gold naacl_format_preds.tsv

    # python evaluator/evaluator.py --predictions data/dev/propara_format_predictions.tsv --answers data/dev/answers.tsv >out.txt
    # mv data/dev/propara_format_predictions.tsv data/dev/propara_format_predictions_action_propara_current_before_combined_${i}.tsv
    filename="out.txt"
    while read -r line; do
      name="$line"
      if echo "$name" | grep 'Overall F1'>o1.txt; then
        set -- $name
        if  [ $(echo "$3 > $best_i_f1" |bc -l) -eq "1" ]; then
          best_i=${i}
          best_i_f1=$3
        fi
      fi
    done < "$filename"
    cd ../pytorch-pretrained-BERT-master
  echo $3
  # fi

done
