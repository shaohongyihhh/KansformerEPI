import pandas as pd
import torch
from glob import glob
from misc_utils import hg19_chromsize

def read_bed(file_path, min_mapped_reads=10, min_percent_methylated=0):
    # 定义列名
    columns = ['chrom', 'start', 'stop', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'mapped_reads', 'percent_methylated']
    
    # 读取文件，并仅保留需要的列和过滤条件
    df = pd.read_csv(file_path, sep='\t', names=columns, usecols=['chrom', 'start', 'stop', 'mapped_reads', 'percent_methylated'])
    df = df.query(f'mapped_reads >= {min_mapped_reads} and percent_methylated > {min_percent_methylated}')
    
    # 过滤掉不在 hg19_chromsize 中的染色体
    df = df[df['chrom'].isin(hg19_chromsize.keys())]
    
    return df

def process_methylation_data(input_folder, output_file, min_mapped_reads=10, min_percent_methylated=0, bin_size=500):
    # 初始化信号字典，按染色体和 bin_size 初始化张量
    chrom_signals = {chrom: torch.zeros((length // bin_size) + 1, dtype=torch.float) for chrom, length in hg19_chromsize.items()}
    
    # 逐个处理 BED 文件
    for file in glob(f'{input_folder}/*.bed.gz'):
        df = read_bed(file, min_mapped_reads, min_percent_methylated)
        
        # 处理每个染色体的区间
        for chrom, start, stop, percent_methylated in df[['chrom', 'start', 'stop', 'percent_methylated']].itertuples(index=False):
            start_bin = start // bin_size
            stop_bin = stop // bin_size
            
            # 向量化处理每个 bin 的最大甲基化百分比
            chrom_signals[chrom][start_bin:stop_bin+1] = torch.max(
                chrom_signals[chrom][start_bin:stop_bin+1], 
                torch.tensor(percent_methylated, dtype=torch.float)
            )
    
    # 保存处理好的张量到指定文件
    torch.save(chrom_signals, output_file)

# 调用主函数
input_folder = '/home/shshao/workplace/KanTransEpi-2/data/methylation_2/K562'
output_file = '/home/shshao/workplace/KanTransEpi-2/data/methylation_2/K562/K562_methy_500bp.pt'
process_methylation_data(input_folder, output_file)
