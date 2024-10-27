sudo apt-get install unzip -y

# Python env setup
conda activate base
conda install -c huggingface transformers huggingface_hub scikit-learn -y
pip install uv && uv pip install wandb pgwalker
# NOTE: wandb login with your api key 

# Dacon
wget https://bit.ly/3gMPScE -O dacon_submit_api-0.0.4-py3-none-any.zip
unzip dacon_submit_api-0.0.4-py3-none-any.zip
pip install dacon_submit_api-0.0.4-py3-none-any.whl
rm -rf dacon_submit_api-0.0.4-py3-none-any.whl dacon_submit_api-0.0.4-py3-none-any.zip