#!/bin/bash
target_dir=results
n_parallel=6
cnt=0
echo "ls -d $target_dir/*.warc.gz.txt"
for file in $(ls -d $target_dir/*.warc.gz.txt); do 
    python extract_possible_sents.py -i $file > $file.extracted 2> $file.log&
    echo $file
    cnt=$((cnt + 1))
    if [ $(($cnt % n_parallel)) = 0 ];then
	wait
	#break
    fi
done

# wait
# for file in $(ls -d $target_dir/*.warc.gz.txt); do 
#     python symbolize_numbers.py -i $file.extracted > $file.symbolized
# done

#rm results/*.log results/*.extracted
