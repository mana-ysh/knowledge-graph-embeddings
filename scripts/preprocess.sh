
cd ../dat

# Freebase
tar zxvf fb15k.tgz
cd FB15k
cat freebase_mtr100_mte100-train.txt | cut -f 2 | sort | uniq > train.rellist
cat freebase_mtr100_mte100-train.txt | cut -f 1,3 | perl -pe 's/\t/\n/g' | sort | uniq > train.entlist
cat freebase_mtr100_mte100-train.txt freebase_mtr100_mte100-valid.txt freebase_mtr100_mte100-test.txt > whole.txt

# WordNet
cd ..
tar zxvf wordnet-mlj12.tar.gz
cd wordnet-mlj12
cat wordnet-mlj12-train.txt | cut -f 2 | perl -pe 's/\t/\n/g' | sort | uniq > train.rellist
cat wordnet-mlj12-train.txt | cut -f 1,3 | perl -pe 's/\t/\n/g' | sort | uniq > train.entlist
cat wordnet-mlj12-train.txt wordnet-mlj12-valid.txt wordnet-mlj12-test.txt > whole.txt
