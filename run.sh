#!/bin/bash
target_dir=results
n_parallel=4
cnt=0
for file in $(ls -d $target_dir/*.warc.gz.txt); do 
    #echo "python main.py -m300000 -i $file > $file.extracted 2> $file.log&"
    #python main.py -m100000 -i $file > $file.extracted 2> $file.log&
    python main.py -i $file > $file.extracted 2> $file.log&
    echo $file
    cnt=$((cnt + 1))
    if [ $(($cnt % n_parallel)) = 0 ];then
	wait
    fi
done
#rm results/*.log results/*.extracted
