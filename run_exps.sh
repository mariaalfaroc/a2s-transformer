#!/bin/bash

for model_type in crnn transformer; do
    for train_ds in Quartets Beethoven Haydn Mozart; do
        if [ $model_type == "transformer" ]; then
            python -u train.py --ds_name $train_ds --model_type $model_type --batch_size 1 --patience 5 --attn_window 100
        else
            python -u train.py --ds_name $train_ds --model_type $model_type --batch_size 1 --patience 5
        fi
        for test_ds in Quartets Beethoven Haydn Mozart; do
            if [ $train_ds != $test_ds ]; then
                python -u test.py --ds_name $test_ds --model_type $model_type --checkpoint_path weights/$model_type/$train_ds.ckpt
            fi
        done
    done
done
