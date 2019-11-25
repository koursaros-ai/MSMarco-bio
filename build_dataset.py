import os
from api import MsMarco
from tqdm import tqdm
import argparse
import random

DATA_DIR = './collectionandqueries'
OUT_DIR = './msmarco-bio'


def main(args):
  bioset = set()

  qrels_tsv_path = os.path.join(OUT_DIR, 'qrels.dev.small.tsv')
  queries_tsv_path = os.path.join(OUT_DIR, 'queries.dev.tsv')

  os.makedirs(OUT_DIR, exist_ok=True)

  print('loading preds..')
  preds_file = os.path.join(args.data_dir, 'preds')
  with open(preds_file) as preds:
    for line in preds:
      pred, doc_id = line.strip().split(' ')
      if float(pred) > 0.5:
        bioset.add(doc_id.strip())

  print('Bioset size %s' % len(bioset))

  queries = dict()
  print('loading queries..')
  query_file_path = os.path.join(args.data_dir, 'queries.train.tsv')
  with open(query_file_path) as query_file:
    for line in query_file:
      qid, query = line.strip().split('\t')
      queries[qid] = query

  qrels = set()
  docs_to_queries = dict()
  qrels_file_path = os.path.join(args.data_dir, 'qrels.train.tsv')
  with open(qrels_file_path) as qrels_file:
    for line in qrels_file:
      qid, _, doc_id, _ = line.strip().split('\t')
      qrels.add((qid, doc_id))
      docs_to_queries[doc_id] = qid

  api = MsMarco('collectionandqueries',host=args.es_host, port=args.es_port, shards=args.shards)

  collection_file = os.path.join(args.data_dir, 'collection.tsv')
  example_num = 0
  qrels_dev_file = open(qrels_tsv_path, 'w')
  queries_dev_file = open(queries_tsv_path, 'w')
  with open(collection_file) as collection, \
          open(os.path.join(args.out_dir, 'collection.tsv'), 'w') as bio, \
          open(os.path.join(args.out_dir, 'triples.train.small.tsv'), 'w') as train:
    with tqdm(total=api.collection_size, desc='BUILDING TRAINING SET') as pbar:
      for line in collection:
        doc_id, text = line.strip().split('\t')
        if doc_id in bioset:
          bio.write(line)
          if doc_id in docs_to_queries:
            qid = docs_to_queries[doc_id]
            example_num += 1
            if example_num < 5000:
              qrels_dev_file.write('\t'.join([str(qid), '0', str(doc_id), '1']) + '\n')
              queries_dev_file.write(qid + '\t' + queries[qid] + '\n')
            else:
              doc_ids, docs = api.es_query(queries[qid], 100)
              negative_example_idx = random.randrange(0, 100)
              # if negative example is the same as the positive, choose the next instead
              if doc_ids[negative_example_idx] == doc_id:
                negative_example_idx += 1
              sample = queries[qid] + '\t' + text + '\t' + docs[negative_example_idx] + '\n'
              train.write(sample)
        pbar.update()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Build training subset')
  parser.add_argument('--es_host', default='localhost')
  parser.add_argument('--es_port', default=9200)
  parser.add_argument('--out_dir', default=OUT_DIR)
  parser.add_argument('--data_dir', default=DATA_DIR)
  parser.add_argument('--shards', default=1)
  args = parser.parse_args()
  main(args)