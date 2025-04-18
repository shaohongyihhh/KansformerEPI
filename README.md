# KansformerEPI: A Deep Learning Framework Integrating KAN and Transformer for Predicting Enhancer-Promoter Interactions

The datasets and codes are available at https://github.com/shaohongyihhh/KansformerEPI.

KansformerEPI is a Integrating KAN and Transformer model for Predicting EPIs

Part of our code is derived from https://github.com/JustlfC03/SCKansformer and https://github.com/biomed-AI/TransEPI. 
We are grateful for their contributions to the source code.

# Requirements

numpy=1.26.2
scikit-learn=1.3.2 
Pytorch=1.10.1
scipy=1.11.3

# Datasets  

All the datasets used in this study are available at (data/BENGI)

# Quickstart

Quick Start Guide for Using the Provided Pre-trained Model. For using other datasets, please refer to the "detailed steps" in the next section.

1.Clone the codes:

git clone https://github.com/shaohongyihhh/KansformerEPI.git 

2.Download processed genomic features

Download the genomic features from the website [https://www.synapse.org/Synapse:syn26156164/wiki/612650].

Methylation data is available in the research by Whalen S et al. on [https://github.com/shwhalen/targetfinder].

3.Run the mode

Use the script to run (pre-configure the dataset to be predicted).
sh ../dev/run_pred.sh

# Detailed steps

## Prepare genomic data 

1.Download the genomic data required by TransEPI from [ENCODE](https://www.encodeproject.org/) or
 [Roadmap](https://egg2.wustl.edu/roadmap/web_portal/processed_data.html#ChipSeq_DNaseSeq)

    - CTCF ChIP-seq data in narrowPeak format
    - DNase-seq data in bigWig format
    - H3K27me3, H3K36me3, H3K4me1, H3K4me3, and H3K9me3 ChIP-seq data in bigWig format

2.Edit "KansformerEPI/data/genomic_data/bed/CTCF_bed.json" and "KansforemrEPI/data/genomic_data/bigwig/bw_6histone.json" 
to specify the location of the narrowPeak and bigWig files 

3.Convert narrowPeak and bigWig signals to `.pt` files 

4.Add the `.pt` files generated by step 3 to "KansformerEPI/data/genomic_data/CTCF_DNase_6histone.500.json"

## Prepare the configuration file for model training

The configuration file should be in `.json` format:

{
    "data_opts": {
        "datasets": [
            "/home/shshao/workplace/KanTransEpi/data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz",
            "/home/shshao/workplace/KanTransEpi/data/BENGI/HeLa.HiC-Benchmark.v3.tsv.gz",
            "/home/shshao/workplace/KanTransEpi/data/BENGI/HeLa.RNAPII-ChIAPET-Benchmark.v3.tsv.gz"
        ],
        "feats_order": ["CTCF", "DNase", "H3K4me1", "H3K4me3", "H3K36me3", "H3K9me3",  "H3K27me3"],
        "feats_config": "/home/shshao/workplace/KanTransEpi/data/genomic_data/CTCF_DNase_6histone.500.json",
        "bin_size": 500,
        "seq_len": 2500000,
        "rand_shift": true
    },

    "model_opts": {
        "model": "KansformerEPI",
        "cnn_channels": [180],
        "cnn_sizes": [11],
        "cnn_pool": [10],
        "enc_layers": 3,
        "num_heads": 6,
        "d_inner": 256,
        "da": 64,
        "r": 32,
        "att_C": 0.1,
        "fc": [128, 64],
        "fc_dropout": 0.2
    },

    "train_opts": {
        "valid_chroms": [],
        "learning_rate": 0.0001,
        "batch_size": 128,
        "num_epoch": 300,
        "patience": 10,
        "num_workers": 16,
        "use_scheduler": false
    }
}

## Prepare input file

The input format should be:

1	36769.5	    chr16	676807	676886	chr16:676807-676886|IMR90|EH37E1139087	chr16	638577	640577	chr16:640076-640077|IMR90|ENSG00000197562.5|ENST00000538492.1|+
0	100424.5	chr16	676807	676886	chr16:676807-676886|IMR90|EH37E1139087	chr16	775771	777771	chr16:777270-777271|IMR90|ENSG00000103253.13|ENST00000389703.3|+
0	101725.5	chr16	676807	676886	chr16:676807-676886|IMR90|EH37E1139087	chr16	777072	779072	chr16:778571-778572|IMR90|ENSG00000103253.13|ENST00000569385.1|+
0	114155.5	chr16	676807	676886	chr16:676807-676886|IMR90|EH37E1139087	chr16	790502	792502	chr16:791001-791002|IMR90|ENSG00000103245.9|ENST00000251588.2|-
0	14966.5  	chr16	676807	676886	chr16:676807-676886|IMR90|EH37E1139087	chr16	690313	692313	chr16:691812-691813|IMR90|ENSG00000172366.15|ENST00000307650.4|+

## Training model

It can be trained by the given [test.ipynb].

# Model

The trained models can be found in [models]

