#!/bin/bash
train_file=$1
model_file=$2
test_file=$3
result_file=$4
k=$5

usage() {
    echo "Usage:$0 train_file model_file test_file result_file k"
    exit 1
}

if [ $# -lt 5 ];then
    usage;
fi

option="-k $k -O 2 "
yakmo $option $train_file $model_file $test_file > $result_file
