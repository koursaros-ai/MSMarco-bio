Building a Custom Search Relevance Eval Set from Open Source Bing Queries
--

There is a notable lack of large scale, easy to use, labeled data sets for information retrieval in most specific domains.

**This repo includes code to generate search queries and passages related to a specific domain or field of knowledge.** We achieve this by selecting a subset of the popular MS Marco dataset. The full MS Marco training set released by Microsoft is much too large to use in it's entirety as a training set.

To build the dataset, clone repo and then run:

`pip install -r requirements.txt`

Examples of queries in the subset generated for health/biology:
- what normal blood pressure by age?
- what is your mandible?
- what part is the sigmoid colon?

Labelling 10k Passages with Google Natural Language API
--

Up to 30k/month document classifications are **free** using <a href = 'https://cloud.google.com/natural-language/'>Google's API</a>. It can be used to classify passages into 700+ categories, and it also reports confidence scores.

**You need to sign up for Google Cloud and authenticate your client first, see https://cloud.google.com/natural-language/docs/reference/libraries**

Then run:

```python
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

SUBSET_SIZE = 10000 # the number of passages to classify

client = language.LanguageServiceClient()

with open('./categories.tsv', 'w+') as outfile:
    with open('./collectionandqueries/collection.tsv') as collection:
        for i, line in enumerate(collection):
            if i > SUBSET_SIZE: break
            try:
                doc_id, doc_text = line.split('\t')
                document = types.Document(
                    content=doc_text,
                    type=enums.Document.Type.PLAIN_TEXT)
                category = client.classify_text(document)
                for cat in category.categories:
                    outfile.write(doc_id+'\t'+cat.name+'\t'+str(cat.confidence)+'\n')
            except: # sometimes the document is too short and the API will err, ignore
                pass
```

Creating a Text Classifier for the Rest of the Set
--

We use <a href='https://github.com/VowpalWabbit/vowpal_wabbit'>vowpal-wabbit</a>(VW) to build a binary text classifier that can classify the rest of the set very fast and for free. Make sure it is installed. (type `vw --help` on the bash). 

Define a function to extract a binary label form the Google NLP Cateogry. In our case we use health/science:

```python
def label_from_category(category, confidence):
    return (1 if 'Health' in category 
    or 'Science' in category else 0, confidence)
```

Then use it to build a VW training set

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

ps = PorterStemmer()

collection_file = './collectionandqueries/collection.tsv'
categories_file = './categories.tsv'
with open(categories_file) as categories:
    categories_dict = dict()
    for line in categories:
        doc_id, category, confidence = line.split('\t')
        categories_dict[doc_id] = label_from_category(category)
        
# input.vw has format <label> <weight> |n <lowercased, stemmed text>
with open('input.vw', 'w') as output, open(collection_file) as collection:
    for line in collection:
        doc_id, text = line.split('\t')
        if doc_id in categories_dict:
            label, confidence = categories_dict[doc_id]
            tokens = [ps.stem(word.lower()) for word in word_tokenize(text)]
            cleaned = re.sub(r'\:', ' ', ' '.join(tokens)) # strip colon bc this is special VW charater
            output.write(str(label)+' '+str(confidence).strip()+' |n '+ cleaned + ' \n')
```

Then train a classifier with this data and save it as bio_model: 

`vw input.vw -f bio_model`

Classifying MSMarco
--

- Download MsMarco collection+queries:
    `wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz`
- Extract:
    `tar -xvzf collectionandqueries.tar.gz`

- Build the porter stmr cli from this repo: https://github.com/wooorm/stmr and make sure it's in your path.

- Run `./classify_msmarco ./collectionandqueries` from MS Marco, producing a file `preds` of format {passage_id} {score}. The higher the score, the more likely it's related to health / bio. 

Building Collection and Queries for the Subset
--

The code for loading the collection and queries from MSMarco and then classifying them is a little more complicated so I won't go over it here. 

If you want to product the set, clone <a href = 'https://github.com/koursaros-ai/MSMarco-bio'>this</a> repo and run the python script:

`python3 build_dataset.py --data_dir <path to collectionsandqueries dir> --out_dir <bio-collectionsandqueries>`

The output folder should contain:

- `collection.tsv`
- `qrels.dev.small.tsv`
- `qrels.train.tsv`
- `queries.dev.small.tsv`
- `queries.train.tsv`

Look <a href = 'https://github.com/microsoft/MSMARCO-Passage-Ranking'>here</a> for more details about the format of these.

Evaluation Results for Bio Subset
--


Pretrained Model  | Finetuning Dataset              | BioMARCO Dev MRR@10 <sup>[1]</sup> |
------------------| --------------------------------| -------------------------------------- |
<a href = 'https://github.com/nyu-dl/dl4marco-bert'>bert-base-uncased-msmarco</a> | MSMarco  | **0.17281** |
<a href = 'https://github.com/naver/biobert-pretrained'>biobert-pubmed-v1.1</a> | MSMarco | 0.17070 | 
BM25 | - | 0.10366

Download dataset <a href='https://storage.googleapis.com/koursaros/bio-collectionandqueries.tar.gz'>here</a> or follow guide below to build it. Look <a href = 'https://github.com/microsoft/MSMARCO-Passage-Ranking'>here</a> for more details about the format of the files.

Check out our main project <a href = 'https://github.com/koursaros-ai/nboost'>NBoost</a> to download and deploy the models with Elasticsearch.

<sup>[1]</sup> Reranking top 50 results from BM25

