# Optional: create and activate a virtual environment
# python -m venv venv && source venv/bin/activate

# Step 1: Install normal dependencies
pip install -r requirements.txt

# Step 2: Install special packages without dependencies
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
pip install --no-deps unsloth

# Step 3: Install from GitHub (transformers and timm upgrades for Gemma 3N)
pip install --no-deps git+https://github.com/huggingface/transformers.git
pip install --no-deps --upgrade timm
