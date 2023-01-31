import os 
from sentence_transformers import LoggingHandler, util
import logging
import tqdm
import pandas as pd
import gzip
import numpy as np

N = 20
N_bias = 12 #equal to 0.6*20 which means lambda=0.6 or 60%
# folder to msmarco-datasets
data_folder = 'msmarco-data'

# file that has the N most biased documents for each query
most_biased_path = 'data/train_run_bias_tf_most_bias_test_tf.txt'
# path to msmarco-QIDPIDTriples train file
train_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.train.tsv.gz')
# path to msmarco-QIDPIDTriples train-eval file
# from https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz
# queries from eval file are not included in the created training set
eval_filepath = os.path.join(data_folder, 'qidpidtriples.rnd-shuf.train-eval.tsv')
# file for the created training set
output_filepath = 'training_set_orig_05.tsv'

if not os.path.exists(train_filepath):
    logging.info("Download "+os.path.basename(train_filepath))
    util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz', train_filepath)

eval = pd.read_csv(eval_filepath, sep='\t', names=['qid', 'pid', 'nid'])
eval_unique_queries = {qid: True for qid in eval.qid.unique().tolist()}

cnt = 0
cnt_pos = 0
cnt_neg = 0
samples_dict = {}

# Randomly sample N entries from trainset
with gzip.open(train_filepath, 'rt') as fIn:
    prev_qid = None
    list_of_lines = []
    i = 0
    for line in tqdm.tqdm(fIn, unit_scale=True):
        qid, pos_id, neg_id = line.strip().split()
        # skip all queries that are inside eval trainset
        if qid in eval_unique_queries:
            continue
        if prev_qid != qid and prev_qid is not None:
            rand_indices = np.random.randint(0, len(list_of_lines), N)
            list_of_sampled_lines = []
            for idx in rand_indices:
                list_of_sampled_lines.append(list_of_lines[idx])
            samples_dict[prev_qid] = list_of_sampled_lines
            list_of_lines = []
        prev_qid = qid
        i+=1
        list_of_lines.append([qid, pos_id, neg_id])

# N_bias = 0 -> completely random sampling
if N_bias != 0:
# overwrite N_bias entries with negative ID's corresponding to the most biased passages
    with open(most_biased_path, 'rt') as fIn:
        prev_qid = None
        list_of_neg_id = []
        for line in tqdm.tqdm(fIn, unit_scale=True):
            qid, neg_id, _ = line.strip().split()
            if qid not in samples_dict:
                continue
            if prev_qid != qid and prev_qid is not None:
                list_of_random_entries = samples_dict[prev_qid]
                for i, neg_id_tmp in enumerate(list_of_neg_id):
                    list_of_random_entries[i][2] = neg_id_tmp
                list_of_neg_id = []
            if len(list_of_neg_id) < N_bias:
                list_of_neg_id.append(neg_id)
            prev_qid = qid


with open(output_filepath, 'w') as fOut:
    for qid in samples_dict:
        list_of_entries = samples_dict[qid]
        for entry in list_of_entries:
            fOut.write(f'{entry[0]}\t{entry[1]}\t{entry[2]}\n')