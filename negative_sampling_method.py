from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
from heapq import nlargest
import logging
import rank_bm25
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import os
import random
import collections
import numpy as np


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


#First, we define the transformer model we want to fine-tune
model_name = 'bert_tiny'
train_batch_size = 32
num_epochs = 1
model_save_path = 'training_ms-marco_cross-encoder-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Negative sampling 
Lambda = 0.6
N = 20


# We train the network with as a binary label task
# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?

# Maximal number of training samples we want to use
max_train_samples = 500000
max_training_queries = 10000

#We set num_labels=1, which predicts a continous score between 0 and 1
model = CrossEncoder(model_name, num_labels=1, max_length=512)


### Now we read the MS Marco dataset
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage

### Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query


def rank_biased_documents(dataset, nb_samples):
    wordlist_path = "/home/claire/PycharmProjects/edds-ex2/bias_metrics/ARaB/resources/wordlist_gender_representative.txt"

    genderwords_feml = []
    genderwords_male = []

    for l in open(wordlist_path):
        vals = l.strip().split(',')
        if vals[1]=='f':
            genderwords_feml.append(vals[0].lower())
        elif vals[1]=='m':
            genderwords_male.append(vals[0].lower())

    genderwords_feml = set(genderwords_feml)
    genderwords_male = set(genderwords_male)

    def get_tokens(text):
      return text.lower().split(" ")

    def get_bias(tokens):
        text_cnt = collections.Counter(tokens)
        
        cnt_feml = 0
        cnt_male = 0
        cnt_logfeml = 0
        cnt_logmale = 0
        for word in text_cnt:
            if word in genderwords_feml:
                cnt_feml += text_cnt[word]
                cnt_logfeml += np.log(text_cnt[word] + 1)
            elif word in genderwords_male:
                cnt_male += text_cnt[word]
                cnt_logmale += np.log(text_cnt[word] + 1)
        text_len = np.sum(list(text_cnt.values()))
        
        bias_tc = (float(cnt_feml - cnt_male), float(cnt_feml), float(cnt_male))
        bias_tf = (np.log(cnt_feml + 1) - np.log(cnt_male + 1), np.log(cnt_feml + 1), np.log(cnt_male + 1))
        bias_bool = (np.sign(cnt_feml) - np.sign(cnt_male), np.sign(cnt_feml), np.sign(cnt_male))
        
        return bias_tc, bias_tf, bias_bool

    docs_bias = {'tc':{}, 'tf':{}, 'bool':{}}
    empty_cnt = 0
    i=0
    for key, value in dataset.items():
        i+=1
        docid = int(key)
        if value != "":
            _text = value
        else:
            _text = ""
            empty_cnt += 1
        
        _res = get_bias(get_tokens(_text))
        docs_bias['tc'][docid] = _res[0]
        docs_bias['tf'][docid] = _res[1]
        docs_bias['bool'][docid] = _res[2]

    # Order documents
    # list of absolute value of the first element of the tf metric (bias)
    v = list(map(abs, list(zip(*docs_bias['tf'].values()))[0]))
    k = list(docs_bias['tf'].keys())    # Order the documents accordingly to their bias
    S_Bq = []
    for _ in range(int(nb_samples)):
        S_Bq += [str(k[v.index(max(v))])]
        v.remove(max(v))
    return set(S_Bq)


def negative_sampling(set_neg_id):
    Lambda = 0.6 # Value of the paper
    N = 20 # Value of the paper
    # for k in range(500000, nb_documents, 500000):
    #  documents = list(corpus.values())[:50000] # reduce the dataset if doesn't fit in RAM
    #  tokenized_corpus_k = [doc.split(" ") for doc in documents]
    #  tokenized_corpus += tokenized_corpus_k
    # bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
    # tokenized_query = query.split(" ")
    # scores = bm25.get_scores(tokenized_query) 
    # ind = np.argpartition(scores, -1000)[-1000:] # n=1000 in the paper
    # D_Mq = {str(i): corpus[str(i)] for i in ind}
    
    D_Mq = {str(neg_id): corpus[neg_id] for neg_id in set_neg_id}
    # Compute the bias with TFARaB select the top lambda * N 
    NS_biased = rank_biased_documents(D_Mq, Lambda * N)
    # NS_rnd select randomply the remaining documents
    NS_rnd = set(random.sample(set_neg_id - NS_biased, int(N - Lambda * N)))
    return NS_biased.union(NS_rnd)


### Creation training & dev data
train_samples = {}
dev_samples = {} # Evaluation during the training
dev = {'positive': set(), 'negative': set()}
train = {'positive': set(), 'negative': set()}

# We use 200 random queries from the train set for evaluation during training
# Each query has at least one relevant and up to 200 irrelevant (negative) passages
num_dev_queries = 200
num_max_dev_negatives = 200 # the negative sample base on the negative sampling.
# msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
# shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
# We extracted in the train-eval split 500 random queries that can be used for evaluation during training

# Read evaluation file
train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    logging.info("Download "+os.path.basename(train_eval_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()
        if qid not in dev and len(dev.keys()) < num_dev_queries + 2:
            dev[qid] = {'positive': set(), 'negative': set()}
        
        if qid in dev:
            dev[qid]['positive'].add(pos_id)
            dev[qid]['negative'].add(neg_id)

for qid in dev.keys():
    if qid != 'positive' and qid != 'negative':
        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': queries[qid], 'positive': set(), 'negative': set()}

        if qid in dev_samples:
            for pos_id in dev[qid]['positive']:
                dev_samples[qid]['positive'].add(corpus[pos_id])

            NS = negative_sampling(dev[qid]['negative'])
            for neg_id in NS:
                dev_samples[qid]['negative'].add(corpus[neg_id])

# Read training file
train_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
if not os.path.exists(train_filepath):
    logging.info("Download "+os.path.basename(train_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)

cnt = 0
with gzip.open(train_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True):
        qid, pos_id, neg_id = line.strip().split()
        if qid in dev_samples:
            continue
        if len(train.keys()) < max_training_queries + 2:
            break
        if qid not in train:
          train[qid] = {'positive': set(), 'negative': set()}

        train[qid]['positive'].add(pos_id)
        train[qid]['negative'].add(neg_id)

for qid in train.keys():
    if qid != 'positive' and qid != 'negative':
        print(train[qid])
        query = queries[qid]
        for pos_id in train[qid]['positive']:
            passage = corpus[pos_id]
            label = 1
            train_samples.append(InputExample(texts=[query, passage], label=label))
        
        NS = negative_sampling(train[qid]['negative'])
        for neg_id in NS:
            passage = corpus[neg_id]
            label = 0
            train_samples.append(InputExample(texts=[query, passage], label=label))
    
        cnt += 1
        if cnt >= max_train_samples:
            break


###  DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision

evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

# Configure the training
warmup_steps = 5000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=10000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=True)

#Save latest model
model.save(model_save_path+'-latest')
