#!/usr/bin/env bash
export DATA_DIR=$1
sed 's/ /|n /' $DATA_DIR/collection.tsv | sed "s/:/ /g" | sed "s/,/ /g" | sed "s/\./ /g" | tr '[:upper:]' '[:lower:]' | stmr | vw -i bio_model --ngram n2 --skips n1 --predictions $DATA_DIR/preds
