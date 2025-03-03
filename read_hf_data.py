'''
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset open-thoughts/OpenThoughts-114k --local-dir ~/liumochi/data/open-thoughts/OpenThoughts-114k
'''
import pprint
from collections import Counter
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, TrainingArguments

## https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
# data_files = '/home/jovyan/liumochi/data/open-thoughts/OpenThoughts-114k/metadata/*'
data_files = "/home/jovyan/liumochi/data/open_thoughts_processed/formatted_test_dataset"

## https://huggingface.co/datasets/AI-MO/NuminaMath-TIR
# data_files = {
#     "train": "/home/jovyan/liumochi/data/AI-MO/NuminaMath-TIR/data/train-00000-of-00001.parquet",
#     "test": "/home/jovyan/liumochi/data/AI-MO/NuminaMath-TIR/data/test-00000-of-00001.parquet"
# }

## https://github.com/Unakar/Logic-RL/tree/main/data/kk/instruct
# data_files = {
#     "train": "/home/jovyan/liumochi/data/Logic-RL/3ppl/train.parquet",
#     "test": "/home/jovyan/liumochi/data/Logic-RL/3ppl/test.parquet"
# }


# dataset = load_dataset("parquet", data_files=data_files)
dataset = load_from_disk(data_files)
print(dataset[0])
# train_dataset = dataset["train"]
# test_dataset = dataset["test"]
# print(train_dataset)
'''
domains = list(dataset["train"]["domain"])
domain_keys = Counter(domains).keys() # equals to list(set(words))
domain_values = Counter(domains).values() # counts the elements' frequency
print(domain_keys)
print(domain_values)
'''
# print(len(train_dataset), len(test_dataset))
# pprint.pprint(peek_dataset, compact=True)

