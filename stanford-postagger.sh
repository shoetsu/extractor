#!/bin/sh
#
# usage: ./stanford-postagger.sh model textFile
#  e.g., ./stanford-postagger.sh models/english-left3words-distsim.tagger sample-input.txt

ROOT_DIR=/home/shoetsu/downloads/stanford-postagger
MODEL_PATH=$ROOT_DIR/models/english-left3words-distsim.tagger
#java -mx300m -cp $ROOT_DIR/stanford-postagger.jar: edu.stanford.nlp.tagger.maxent.MaxentTagger -model $MODEL_PATH -textFile $1 -outputFormat tsv -sentenceDelimiter newline -tokenize false > $1.tagged
java -mx2g -cp $ROOT_DIR/stanford-postagger.jar: edu.stanford.nlp.tagger.maxent.MaxentTagger -model $MODEL_PATH -textFile $1 -outputFormat tsv -sentenceDelimiter newline -tokenize false > $1.tagged

#echo "java -mx300m -cp $ROOT_DIR/stanford-postagger.jar: edu.stanford.nlp.tagger.maxent.MaxentTagger -model $MODEL_PATH -textFile $1 -outputFormat tsv -sentenceDelimiter newline -tokenize false > $1.tagged"










