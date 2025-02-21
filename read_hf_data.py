'''
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset open-thoughts/OpenThoughts-114k --local-dir ~/liumochi/data/open-thoughts/OpenThoughts-114k
'''

from collections import Counter
from datasets import load_dataset

# https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
data_dir = '/home/jovyan/liumochi/data/open-thoughts/OpenThoughts-114k/metadata/*'

# data_files = {
#     "train": "/home/jovyan/liumochi/data/AI-MO/NuminaMath-TIR/data/train-00000-of-00001.parquet",
#     "test": "/home/jovyan/liumochi/data/AI-MO/NuminaMath-TIR/data/test-00000-of-00001.parquet"
# }

dataset = load_dataset("parquet", data_files=data_dir)
peek_dataset = dataset["train"][0]
domains = list(dataset["train"]["domain"])

domain_keys = Counter(domains).keys() # equals to list(set(words))
domain_values = Counter(domains).values() # counts the elements' frequency
print(domain_keys)
print(domain_values)