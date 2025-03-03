if [ $# -eq 2 ]; then
  REPO_TYPE=$1   # model or data
  ITEM_NAME=$2   # Qwen/Qwen2.5-7B-Instruct

  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download --repo-type $REPO_TYPE $ITEM_NAME --local-dir ~/liumochi/$REPO_TYPE/$ITEM_NAME
else
  echo "$0 <REPO_TYPE: model/data> <ITEM_NAME: Qwen/Qwen2.5-7B-Instruct>"
fi