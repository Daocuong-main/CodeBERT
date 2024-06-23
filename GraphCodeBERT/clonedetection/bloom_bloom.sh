mkdir blom_blom
python run_print.py \
    --output_dir=blom_blom \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --do_eval \
    --train_data_file=dataset/Multiclass/train_100.txt \
    --eval_data_file=dataset/Multiclass/test_100.txt \
    --test_data_file=dataset/Multiclass/test_100.txt \
    --epoch 1 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee blom_blom/blom_blom.log
# python run.py \
#     --output_dir=saved_models_50_epoch_5000_samples \
#     --config_name=microsoft/graphcodebert-base \
#     --model_name_or_path=microsoft/graphcodebert-base \
#     --tokenizer_name=microsoft/graphcodebert-base \
#     --do_eval \
#     --train_data_file=dataset/Multiclass/train_5000.txt \
#     --eval_data_file=dataset/Multiclass/test_5000.txt \
#     --test_data_file=dataset/Multiclass/test_5000.txt \
#     --epoch 1 \
#     --code_length 512 \
#     --data_flow_length 128 \
#     --train_batch_size 4 \
#     --eval_batch_size 4 \
#     --learning_rate 2e-5 \
#     --max_grad_norm 1.0 \
#     --evaluate_during_training \
#     --seed 123456 2>&1| tee saved_models_50_epoch_5000_samples/eval_5000_samples.log