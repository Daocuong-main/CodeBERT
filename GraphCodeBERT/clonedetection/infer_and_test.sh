python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=dataset/Multiclass/train.txt \
    --eval_data_file=dataset/Multiclass/test.txt \
    --test_data_file=dataset/Multiclass/test.txt \
    --epoch 1 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/test.log
# python evaluator/evaluator.py -a dataset/Multiclass/test.txt -p saved_models/predictions.txt 2>&1| tee saved_models/score.log