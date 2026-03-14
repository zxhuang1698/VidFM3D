# create a virtual env and activate (conda as an example)
conda create -n opensora python=3.10
conda activate opensora

# Ensure torch >= 2.4.0
pip install -v . # for development mode, `pip install -v -e .`
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 # install xformers according to your cuda version
pip install flash-attn --no-build-isolation

cd TensorNVMe
pip install -v --no-cache-dir .
pip install "huggingface_hub[cli]"
cd ..
