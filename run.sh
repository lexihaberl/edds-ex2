#!/bin/bash
# python3 reranker.py\
#      -checkpoint /home/dalina/David/Uni/edds/edds-ex2/output/training_ms-marco_cross-encoder-prajjwal1-bert-tiny-2023-01-29_20-06-46-latest/ \
#      -queries QS2.tsv\
#      -run /home/dalina/David/Uni/edds/edds-ex2/trec_runs/215_neutral_queries/bert_tiny/ranked_list_original.trec \
#      -res res.txt


#RERANKER
MODEL_CHECKPOINT=/home/dalina/David/Uni/edds/edds-ex2/output/training_ms-marco_cross-encoder-prajjwal1-bert-tiny-2023-01-29_22-33-59not_aware-latest
QUERIES=QS2.tsv
RUN=/home/dalina/David/Uni/edds/edds-ex2/QS2_BM25.tsv
RERANKER=res/reranker_tiny_not_aware_paper.trec

#MMR
MMR=res/resMrr_tiny_not_aware_paper.txt

#ARaB
ROOT_PATH="../../trec_runs/215_neutral_queries/bert_tiny/"
EXPERIMENT="../../res/reranker_tiny_not_aware_paper.trec"

python3 reranker.py\
     -checkpoint $MODEL_CHECKPOINT \
     -queries $QUERIES \
     -run $RUN \
     -res $RERANKER

python3 MRR_calculator.py \
     -run $RERANKER \
     -result $MMR

python3 bias_metrics/ARaB/runs_calculate_bias.py \
     -root_path $ROOT_PATH \
     -experiment $EXPERIMENT

python3 bias_metrics/ARaB/model_calculate_bias.py \
     -experiment $EXPERIMENT

python3 bias_metrics/NFaiRR/metrics_fairness.py \
     --backgroundrunfile $RUN \
     --runfile $RERANKER \
     --experiment $EXPERIMENT