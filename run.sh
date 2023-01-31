#!/bin/bash
# python3 reranker.py\
#      -checkpoint /home/dalina/David/Uni/edds/edds-ex2/output/training_ms-marco_cross-encoder-prajjwal1-bert-tiny-2023-01-29_20-06-46-latest/ \
#      -queries QS2.tsv\
#      -run /home/dalina/David/Uni/edds/edds-ex2/trec_runs/215_neutral_queries/bert_tiny/ranked_list_original.trec \
#      -res res.txt


#RERANKER
# trained model that should be evaluated
MODEL_CHECKPOINT=/home/dalina/David/Uni/edds/edds-ex2/output/prajjwal1-bert-tiny-2023-01-30_15-58-12aware_01-latest
# query set to be evaluated on
QUERIES=QS2.tsv
# run file of the top k queries from BM25 for the QUERIES dataset
RUN=/home/dalina/David/Uni/edds/edds-ex2/QS2_BM25.tsv
# save path of result file
RERANKER=res/reranker_aware_01.trec

#MMR
# save path of result file
MMR=res/resMrr_aware_01.txt

#ARaB
# irrelevant
ROOT_PATH="../../trec_runs/215_neutral_queries/bert_tiny/"
# runfile of the reranked documents. Should be the same file as RERANKER
EXPERIMENT="../../res/reranker_aware_01.trec"

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