MSMarco-Bio
--

There is a notable lack of large scale, easy to use, labeled data sets for information retrieval in health / medicine.

However, Microsoft published a **huge** collection of bing queries and corresponding correct search results, many of which, it turns out, are related to health, medicine and biology.

This repo includes code to generate a subset of **75,000 MS Marco queries related to health**, but it could easily be adapted for any specific domain.

Clone repo and then run:

`pip install -r requirements.txt`

Baseline Results for BioMarco
--

Pretrained Model  | Finetuned on | Dev MRR @10 |
------------------| ------------ | ------------|
bert-base-uncased | MsMarco small| .21         |

Labelling 10k Passages with Google Natural Language API
--

Up to 30k/month document classifications are **free** using <a href = 'https://cloud.google.com/natural-language/'>Google's API</a>. It classifies any text into 700+ categories, and it also reports confidence scores.

**You need to sign up for google cloud and authenticate your client first, see https://cloud.google.com/natural-language/docs/reference/libraries**

Then run:

```python
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

client = language.LanguageServiceClient()

with open('./categories.tsv', 'w+') as outfile:
    with open('./collectionandqueries/collection.tsv') as collection:
        for i, line in enumerate(collection):
            if i > 10000: break
            try:
                doc_id, doc_text = line.split('\t')
                document = types.Document(
                    content=doc_text,
                    type=enums.Document.Type.PLAIN_TEXT)
                category = client.classify_text(document)
                for cat in category.categories:
                    outfile.write(doc_id+'\t'+cat.name+'\t'+str(cat.confidence)+'\n')
            except: # sometimes the document is too short and the API with err, ignore
                pass
```

Creating a Faster Text Classifier with Vowpal Wabbit
--

We use <a href='https://github.com/VowpalWabbit/vowpal_wabbit'>vowpal-wabbit</a> to build a binary text classifier that can classify the entire rest of the set very fast and for free. Make sure it is installed. (type `vw --help` on the bash). 

### Build a training set for a health related classifier

Define a function to extract a binary label form the Google NLP Cateogry. In our case we use health/ science.

```python
def label_from_category(category, confidence):
    return (1 if 'Health' in category 
    or 'Science' in category else 0, confidence)

## Then use it to build a VW traininig set
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

**Note: I don't use python for this process because it's much slower**

- Download MsMarco collection+queries:
    `wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz`
- Extract:
    `tar -xvzf collectionandqueries.tar.gz`


Build the porter stmr cli from this repo: https://github.com/wooorm/stmr and make sure it's in your path.

Classify the entire `collections.tsv` from MS Marco, producing a file `preds` of format {passage_id} {score}. The higher the score, the more likely it's related to health/ bio.
`./run.sh`
or 

**NOTE: The tab in the first sed often turns into a space if you copy/ paste. You need to press ctrl-v on the bash and then tab to replace it. or just run the run.sh script**
```bash
export DATA_DIR=collectionandqueries
sed 's/ /|n /' collectionandqueries/collection.tsv \
| sed "s/:/ /g" | sed "s/,/ /g" | sed "s/\./ /g" | \
tr '[:upper:]' '[:lower:]' | stmr | \
vw -i bio_model --ngram n2 --skips n1 --predictions $DATA_DIR/preds
```

Building Collection and Queries for the Subset
--

<a href = 'https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started.html'>Set up an Elasticsearch instance</a> if you don't already have one, we need this to produce negative training examples using BM25.

Build a training set of the triples of form 
`{query}    {positive example}  {negative example}`

And collection.tsv, queries.dev.tsv and qrels.dev.small.tsv of the same form as the original MS Marco dataset.

`python3 build_dataset.py --es_host <elasticsearch host> --es_port <es port> --out_dir <out_dir>...`




