import argparse
import ir_datasets

def main():

    parser = argparse.ArgumentParser()
    #parser.add_argument('-qrels', type=str, default='')
    parser.add_argument('-run', type=str, default='')
    parser.add_argument('-metric', type=str, default='mrr_cut_10')
    parser.add_argument('-result', type=str, default='')
    args = parser.parse_args()

    metric = args.metric
    k = int(metric.split('_')[-1])

    dataset = ir_datasets.load("msmarco-passage/dev/judged")
    qrel = {}
    for qrel_vals in dataset.qrels_iter():
        qid, did, relevance, label = qrel_vals
        if qid not in qrel:
            qrel[qid] = {}
        qrel[qid][did] = int(relevance)
    # qrel = {}
    # with open(args.qrels, 'r') as f_qrel:
    #     for line in f_qrel:
    #         qid, _, did, label = line.strip().split()
    #         if qid not in qrel:
    #             qrel[qid] = {}
    #         qrel[qid][did] = int(label)
    run = {}
    with open(args.run, 'r') as f_run:
        for line in f_run:
            vals = line.strip().split()
            if len(vals) == 6:
                qid, _, did, _, _, _ = line.strip().split()
            elif len(vals) == 3:
                qid, did, _ = line.strip().split()
            if qid not in run: 
                run[qid] = []
            run[qid].append(did)
    if (len(args.result) != 0):
        myfile = open(args.result, "w") 

    mrr = 0.0
    for qid in run:
        rr = 0.0
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                rr = 1 / (i+1)
                break
        if (len(args.result) != 0):
            myfile.write(str(qid) + "\t" + str(rr) + "\n")
        mrr += rr
    mrr /= len(run)
    if (len(args.result) != 0):
        myfile.write(str("total") + "\t" + str(mrr) + "\t")
        myfile.close()
    print("MRR@10: ", mrr)
    
    with open('res/metrics_'+args.run[13:-5] + '.txt', 'a+') as fss:
        fss.write("MRR@10: " + str(mrr)+'\n')


if __name__ == "__main__":
    main()
