## 不同版本的训练和验证json文件的特征以及比较好的实验结果总结

- train_jianjie
- train_jianjie2:目前已经跑通过一次的版本，里面的数据依照时间间隔3之差值来构造，低于1.0的差值直接忽略，label为某个时刻的label
- train_jinanjie3：里面的数据依照时间间隔6（因为故障的持续时间为5分钟）的差值来构造，低于2.0的差值直接忽略，prompt去掉了前面冗余的话，直接从数值比较开始。训练数据按照abnormal:noraml=1:8的比例进行组织，其中abnormal一共904条数据，所有数据一共8136条。label为某个时刻的label某个节点的label，并且验证集不能随机划分必须按照一定的顺序，也就是说相当于6个要为1组。dev_jianjie3没有去掉冗余的话，且差值阈值为1，dev_jianjie3_2去掉了冗余的话，且差值阈值为2
- train_jianjie4:与3不同的是进行了特征裁剪，同时指标名和值之间用冒号分割，3后续可以尝试一下将值和指标名用冒号分割
- jianjie3_rootcause_analyze文件夹是将jianjie3文件夹下的train_jianjie3.json和dev_jianjie3_2.json中波动较大的异常指标作为根因分析加入到summary字段中，形成dev_jianjie3_2_final.json和train_jianjie3_final.json，目前还没有设置波动阈值，这个文件夹下还有一些其他文件例如rca相关的，是将所以异常数据拿出来进行简单分析的中间文件。prompt.py记录了即将传入大模型的prompt

目前比较好的实验效果：
1. jinajie3/train_jianjie3，模型路径为ptuning/output/aiops-chatglm-6b-pt-128-2e-2-0628/checkpoint-1000，bash.sh参数如下

PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file AIOPS/jianjie3/train_jianjie3.json \
    --validation_file AIOPS/jianjie3/dev_jianjie3.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir output/aiops-chatglm-6b-pt-$PRE_SEQ_LEN-$LR-0628 \
    --overwrite_output_dir \
    --max_source_length 400 \
    --max_target_length 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

目前在dev+jianjie3_2.json的前1000条数据中的测试结果为(测试代码为chatglm/test.py)：
- 单节点评估，label修正前：
accuracy:0.97
precision:0.50
recall:0.94
              precision    recall  f1-score   support

           0       1.00      0.97      0.98       969
           1       0.50      0.94      0.65        33

    accuracy                           0.97      1002
   macro avg       0.75      0.95      0.82      1002
weighted avg       0.98      0.97      0.97      1002

- 聚合节点评估，label修正前:
(聚合节点的评估代码均在对应数据文件夹下analyze.py文件中，例如jianjie3/analyze.py)
accuracy:0.82
precision:0.77
recall:0.98
report:
              precision    recall  f1-score   support

           0       0.96      0.63      0.76       440
           1       0.77      0.98      0.86       545

    accuracy                           0.82       985
   macro avg       0.86      0.80      0.81       985
weighted avg       0.85      0.82      0.82       985

- 单节点评估，label修正后:
(label修正相关在jianji3/analyze.py和result.txt中)：

accuracy:0.98
precision:0.69
recall:0.98
report:
              precision    recall  f1-score   support

           0       1.00      0.98      0.99       958
           1       0.69      0.98      0.81        44

    accuracy                           0.98      1002
   macro avg       0.85      0.98      0.90      1002
weighted avg       0.99      0.98      0.98      1002

- 聚合节点评估，label修正后:

accuracy:0.94
precision:0.94
recall:0.98
report:
              precision    recall  f1-score   support

           0       0.96      0.86      0.91       322
           1       0.94      0.98      0.96       663

    accuracy                           0.94       985
   macro avg       0.95      0.92      0.93       985
weighted avg       0.94      0.94      0.94       985



- bert的单节点最好指标(label修正前)
              precision    recall  f1-score   support

           0       1.00      0.98      0.99       967
           1       0.63      0.88      0.73        33

    accuracy                           0.98      1000
   macro avg       0.81      0.93      0.86      1000
weighted avg       0.98      0.98      0.98      1000

[[950   4]
 [ 17  29]]

 - bert的多节点最好指标(label修正前)
              precision    recall  f1-score   support

           0       0.90      0.74      0.81       438
           1       0.82      0.94      0.87       545

    accuracy                           0.85       983
   macro avg       0.86      0.84      0.84       983
weighted avg       0.86      0.85      0.85       983

[[323  34]
 [115 511]]