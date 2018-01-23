#!/bin/bash
#
# usage: ./stanford-postagger.sh model textFile
#  e.g., ./stanford-postagger.sh models/english-left3words-distsim.tagger sample-input.txt

usage() {
    echo "Usage:$0 input_file"
    exit 1
}

if [ $# -lt 1 ];then
    usage;
fi

if [ $# -lt 2 ];then
    memory=2g
else
    memory=$2
fi

input_path=$1
ROOT_DIR=/home/shoetsu/downloads/stanford-postagger
MODEL_PATH=$ROOT_DIR/models/english-left3words-distsim.tagger
#java -mx300m -cp $ROOT_DIR/stanford-postagger.jar: edu.stanford.nlp.tagger.maxent.MaxentTagger -model $MODEL_PATH -textFile $1 -outputFormat tsv -sentenceDelimiter newline -tokenize false > $1.tagged
java -mx$memory -cp $ROOT_DIR/stanford-postagger.jar: edu.stanford.nlp.tagger.maxent.MaxentTagger -model $MODEL_PATH -textFile $input_path -outputFormat tsv -sentenceDelimiter newline -tokenize false > $input_path.tagged

#echo "java -mx300m -cp $ROOT_DIR/stanford-postagger.jar: edu.stanford.nlp.tagger.maxent.MaxentTagger -model $MODEL_PATH -textFile $1 -outputFormat tsv -sentenceDelimiter newline -tokenize false > $1.tagged"










