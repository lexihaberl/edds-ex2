import collections
from re import M
import numpy as np
import pickle
import pandas as pd


docs_bias_paths = {'tc':"data/msmarco_passage_docs_bias_tc.pkl",
                        'bool':"data/msmarco_passage_docs_bias_bool.pkl",
                        'tf':"data/msmarco_passage_docs_bias_tf.pkl"}

at_ranklist = [5, 10, 20, 30, 50, 100]

root_path = "../../trec_runs/215_neutral_queries/bert_tiny/"
# the path of these run files should be set
experiments = {'run_file_biased': root_path + "ranked_list_original.trec",
                'run_file_unbiased': root_path + 'ranked_list_fairness_aware.trec',
               }
experiments = {'run_file_biased': root_path + "ranked_list_original.trec",
                 'run_file_unbiased': "../../res.txt",
                }
experiments = {'train': '../../anserini/runs/run.msmarco-passage.train.tsv'}
docs_bias_paths = {'tf':"data/msmarco_passage_docs_bias_tf.pkl"}
at_ranklist = [10]
#Loading saved document bias values
docs_bias = {}

runs_docs_bias = {}
for exp_name in experiments:
    runs_docs_bias[exp_name] = {}

for _method in docs_bias_paths:
    print (_method)
    with open(docs_bias_paths[_method], 'rb') as fr:
        docs_bias[_method] = pickle.load(fr)

    for exp_name in experiments:
        print (exp_name)

        run_path = experiments[exp_name]
        
        runs_docs_bias[exp_name][_method] = {}
        most_biased_docs = {}
        
        with open(run_path) as fr:
            qryid_cur = 0
            prev_qry_id = None
            map_of_docs = {}
            for i, line in enumerate(fr):
                if (i % 5000000 == 0) and (i != 0):
                    print ('line', i)

                vals = line.strip().split('\t')
                if len(vals) == 3:
                    qryid = int(vals[0])
                    docid = int(vals[1])
                    rank = int(vals[2])
                    if prev_qry_id is not None and prev_qry_id != qryid:
                        most_biased_docs[prev_qry_id] = dict(sorted(map_of_docs.items(), key = lambda x: x[1], reverse = True)[:20])
                        map_of_docs = {}
                    prev_qry_id = qryid
                    bias_metric, _, _ = docs_bias[_method][docid]
                    map_of_docs[docid] = abs(bias_metric)
    docs_bias[_method] = None
    print ('done!')


for exp_name in experiments:
    for _method in docs_bias_paths:
        save_path = "data/%s_run_bias_%s" % (exp_name, _method)

        print (save_path)

        with open(save_path + '_most_bias_test_tf.txt', 'w') as fw:
            for qry in most_biased_docs.keys():
                for doc in most_biased_docs[qry].keys():
                    fw.write(f'{qry}\t{doc}\t{most_biased_docs[qry][doc]}\n')
