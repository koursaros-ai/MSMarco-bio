import os
from tqdm import tqdm
import argparse


def load_queries(set_name):
  queries = dict()
  print('loading %s queries..' % set_name)
  query_file_path = os.path.join(args.data_dir, 'queries.%s.tsv' % set_name)
  with open(query_file_path) as query_file:
    for line in query_file:
      qid, query = line.strip().split('\t')
      queries[qid] = query
  return queries


def load_qrels(set_name):
  qrels = set()
  docs_to_queries = dict()
  qrels_file_path = os.path.join(args.data_dir, 'qrels.%s.tsv' % set_name)
  with open(qrels_file_path) as qrels_file:
    for line in qrels_file:
      qid, _, doc_id, _ = line.strip().split('\t')
      qrels.add((qid, doc_id))
      docs_to_queries[doc_id] = qid
  return qrels, docs_to_queries


def main(args):
  subset = set()
  os.makedirs(args.out_dir, exist_ok=True)

  print('loading preds..')
  with open(args.preds_file) as preds:
    for line in preds:
      pred, doc_id = line.strip().split(' ')
      if float(pred) > 0.5:
        subset.add(doc_id)

  print('Subset size is %s passages' % len(subset))

  collection_file = os.path.join(args.data_dir, 'collection.tsv')
  collection_size = len([' ' for _ in open(collection_file)])

  qrels_map = dict()
  queries_map = dict()
  qrels_files = dict()
  queries_files = dict()

  sets = ['train', 'dev.small']
  for set_name in sets:
    qrels_map[set_name] = load_qrels(set_name)
    queries_map[set_name] = load_queries(set_name)
    qrels_files[set_name] = open(os.path.join(args.out_dir, 'qrels.%s.tsv' % set_name), 'w')
    queries_files[set_name] = open(os.path.join(args.out_dir, 'queries.%s.tsv' % set_name), 'w')

  example_num = 0
  with open(os.path.join(args.out_dir, 'collection.tsv'), 'w') as subset_collection:
    with open(collection_file) as collection:
      with tqdm(total=collection_size, desc='BUILDING %s SETS' % (','.join(sets))) as pbar:
        for line in collection:
          doc_id, text = line.strip().split('\t')
          if doc_id in subset:
            subset_collection.write(line)
            for split in sets:
              qrels, docs_to_queries = qrels_map[split]
              queries = queries_map[split]
              if doc_id in docs_to_queries:
                qid = docs_to_queries[doc_id]
                example_num += 1
                qrels_files[split].write('\t'.join([str(qid), '0', str(doc_id), '1']) + '\n')
                queries_files[split].write(qid + '\t' + queries[qid] + '\n')
          pbar.update()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Build training subset')
  parser.add_argument('--data_dir', default='./collectionandqueries')
  parser.add_argument('--out_dir', default='./collectionandqueries-subset')
  parser.add_argument('--preds_file', default='./preds')
  args = parser.parse_args()
  main(args)