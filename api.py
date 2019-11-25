import elasticsearch
import csv
from pathlib import Path
from elasticsearch import Elasticsearch
from nboost.base.helpers import es_bulk_index
from tqdm import tqdm

REQUEST_TIMEOUT = 10000


class MsMarco:
    """MSMARCO dataset builder"""

    def __init__(self, data_dir, shards = 1, host='localhost', port=9200):

        self.dataset_dir = Path(data_dir)
        self.qrels_tsv_path = self.dataset_dir.joinpath('qrels.dev.small.tsv')
        self.queries_tsv_path = self.dataset_dir.joinpath('queries.dev.tsv')
        self.collections_tsv_path = self.dataset_dir.joinpath('collection.tsv')
        self.index = 'ms_marco' # self.dataset_dir.name
        self.host = host
        self.port = port
        self.qrels = set()
        self.queries = dict()

        self.es = Elasticsearch(
            host=self.host,
            port=self.port,
            timeout=REQUEST_TIMEOUT)

        self.collection_size = 0
        with open(str(self.collections_tsv_path)) as collection:
            for _ in collection: self.collection_size += 1

        # INDEX MSMARCO
        try:
            if self.es.count(index=self.index)['count'] < self.collection_size:
                raise elasticsearch.exceptions.NotFoundError
        except elasticsearch.exceptions.NotFoundError:
            try:
                self.es.indices.create(index=self.index, body={
                    'settings': {
                        'index': {
                            'number_of_shards': shards
                        }
                    }
                })
            except: pass
            print('Indexing %s' % self.collections_tsv_path)
            es_bulk_index(self.es, self.stream_msmarco_full())

        print('Reading %s' % self.qrels_tsv_path)
        with self.qrels_tsv_path.open() as file:
            qrels = csv.reader(file, delimiter='\t')
            for qid, _, doc_id, _ in qrels:
                self.qrels.add((qid, doc_id))

        print('Reading %s' % self.queries_tsv_path)
        with self.queries_tsv_path.open() as file:
            queries = csv.reader(file, delimiter='\t')
            for qid, query in queries:
                self.queries[qid] = query

    def stream_msmarco_full(self):
        print('Optimizing streamer...')
        num_lines = sum(1 for _ in self.collections_tsv_path.open())
        with self.collections_tsv_path.open() as fh:
            data = csv.reader(fh, delimiter='\t')
            with tqdm(total=num_lines, desc='INDEXING MSMARCO') as pbar:
                for ident, passage in data:
                    body = dict(_index=self.index,
                                _id=ident, _source={'passage': passage})
                    yield body
                    pbar.update()

    def es_query(self, query: str, topk: int):
        body = dict(
            size=topk,
            query={"match": {"passage": {"query": query}}})

        res = self.es.search(
            index=self.index,
            body=body,
            filter_path=['hits.hits._*'])

        doc_ids = [hit['_id'] for hit in res['hits']['hits']]
        docs = [hit['_source']['passage'] for hit in res['hits']['hits']]
        return doc_ids, docs


