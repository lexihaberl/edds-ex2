import collections
import numpy as np
import pickle
import ir_datasets

# dataset_path = '/home/dalina/David/Uni/edds/edds-ex2/msmarco-data/collection.tsv'
# dataset = pd.read_csv(dataset_path,sep = '\t')

dataset = ir_datasets.load("msmarco-passage/dev/small")
wordlist_path = "resources/wordlist_gender_representative.txt"

docs_bias_save_paths = {'tc':"data/msmarco_passage_docs_bias_tc.pkl",
                        'bool':"data/msmarco_passage_docs_bias_bool.pkl",
                        'tf':"data/msmarco_passage_docs_bias_tf.pkl"}

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

print(len(genderwords_feml), len(genderwords_male))

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
for i, vals in enumerate(dataset.docs_iter()):
    docid = int(vals[0])
    if len(vals) == 2:
        _text = vals[1]
    else:
        _text = ""
        empty_cnt += 1
    
    _res = get_bias(get_tokens(_text))
    docs_bias['tc'][docid] = _res[0]
    docs_bias['tf'][docid] = _res[1]
    docs_bias['bool'][docid] = _res[2]
        
    if i % 1000000 == 0:
        print (i)
            
print ('done!')
print ('number of skipped documents: %d' % empty_cnt)

# saving bias values of documents
for _method in docs_bias:
    print (_method)
    with open(docs_bias_save_paths[_method], 'wb') as fw:
        pickle.dump(docs_bias[_method], fw)
    print (_method + ' done')
