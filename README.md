'''
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
'''

# Extract all possible sentence from $corpus_dir/*.warc.gz.txt and filter them by some rules. (containing words with CD(cardinal digits) POS or  )
1. ./extract_possible_sents.sh $corpus_dir  

# Put all the extracted files into together.
2. cat $corpus_dir/*.warc.gz.txt.extracted > all.extracted

# Tokenize them and normalize numbers.
3. python tokenize_and_normalize -i all.extracted > all.normalized

# To get rid of noisy ones, apply more strict rules.
4. python apply_strict_rules.py -i all.normalized > all.normalized.strict

5. python random_pickup.py > all.normalized.strict.10000
