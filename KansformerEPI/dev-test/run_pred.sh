#!/bin/sh

fold0="chr1 chr10 chr15 chr21"
fold1="chr19 chr3 chr4 chr7 chrX"
fold2="chr13 chr17 chr2 chr22 chr9"
fold3="chr12 chr14 chr16 chr18 chr20"
fold4="chr11 chr5 chr6 chr8"

for i in $(seq 1 30); do
    # Create a new directory for each iteration
    mkdir -p /home/shshao/workplace/KansformerEPI/dev-test/test_${i}


    for fn in /home/shshao/workplace/KansformerEPI/data/BENGI/HMEC.HiC-Benchmark.v3.tsv.gz /home/shshao/workplace/KansformerEPI/data/BENGI/NHEK.HiC-Benchmark.v3.tsv.gz /home/shshao/workplace/KansformerEPI/data/BENGI/K562.HiC-Benchmark.v3.tsv.gz /home/shshao/workplace/KansformerEPI/data/BENGI/IMR90.HiC-Benchmark.v3.tsv.gz; do
        echo $fn
        bn=`basename $fn .tsv`
        for fold in 0 1 2 3 4; do
            echo "- Fold $fold"
            if   [ $fold -eq 0 ]; then
                /home/shshao/workplace/KansformerEPI/src/evaluate_model.py --batch-size 64 -t $fn -c "/home/shshao/workplace/KansformerEPI/dev-test/config_500bp.json" --test-chroms $fold0 -m /home/shshao/workplace/KansformerEPI/output/checkpoint_fold0.best_epoch1.pt -p /home/shshao/workplace/KansformerEPI/dev-test/test_${i}/results_${bn}.fold${fold}_strict
            elif [ $fold -eq 1 ]; then
                /home/shshao/workplace/KansformerEPI/src/evaluate_model.py --batch-size 64 -t $fn -c "/home/shshao/workplace/KansformerEPI/dev-test/config_500bp.json" --test-chroms $fold1 -m /home/shshao/workplace/KansformerEPI/output/checkpoint_fold1.best_epoch1.pt -p /home/shshao/workplace/KansformerEPI/dev-test/test_${i}/results_${bn}.fold${fold}_strict
            elif [ $fold -eq 2 ]; then
                /home/shshao/workplace/KansformerEPI/src/evaluate_model.py --batch-size 64 -t $fn -c "/home/shshao/workplace/KansformerEPI/dev-test/config_500bp.json" --test-chroms $fold2 -m /home/shshao/workplace/KansformerEPI/output/checkpoint_fold2.best_epoch1.pt -p /home/shshao/workplace/KansformerEPI/dev-test/test_${i}/results_${bn}.fold${fold}_strict
            elif [ $fold -eq 3 ]; then
                /home/shshao/workplace/KansformerEPI/src/evaluate_model.py --batch-size 64 -t $fn -c "/home/shshao/workplace/KansformerEPI/dev-test/config_500bp.json" --test-chroms $fold3 -m /home/shshao/workplace/KansformerEPI/output/checkpoint_fold3.best_epoch1.pt -p /home/shshao/workplace/KansformerEPI/dev-test/test_${i}/results_${bn}.fold${fold}_strict
            elif [ $fold -eq 4 ]; then
                /home/shshao/workplace/KansformerEPI/src/evaluate_model.py --batch-size 64 -t $fn -c "/home/shshao/workplace/KansformerEPI/dev-test/config_500bp.json" --test-chroms $fold4 -m /home/shshao/workplace/KansformerEPI/output/checkpoint_fold4.best_epoch1.pt -p /home/shshao/workplace/KansformerEPI/dev-test/test_${i}/results_${bn}.fold${fold}_strict
            fi

        done &> /home/shshao/workplace/KansformerEPI/dev-test/test_${i}/results_${bn}_strict.log &
        sleep 20
    done
    wait
done


