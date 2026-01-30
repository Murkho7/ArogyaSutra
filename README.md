<h2 align="center">ArogyaSutra: A Multi-Agent Framework for Multimodal Medical Reasoning in
Indic Languages
</h2>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2f7995eb-65b6-4c36-9af5-848ce3cb2d4d"
       alt="AragyaSutra"
       width="300" />
</p>
<p align="center">
  📃 <a href="#"><b>Paper@</b></a> |
  🤗 <a href="https://huggingface.co/collections/iit-patna-cse-ai/arogyasutra"><b>Models & Datasets Repo*</b></a>
</p>

<p align="center">
  <b>
  *A Preview of our dataset is given at Hugging Face. click on "Models & Datasets Repo" <br/>
  @ Paper and code will be uploaded Soon.
</b>
</p>

<p align="left">

Existing MLLMs, predominantly trained on English-centric data, struggle to support such use cases, limiting equitable access to AI-driven healthcare assistance. To address this challenge, we construct a large-scale multilingual multimodal medical question–answer dataset from eight heterogeneous sources, covering 31 body systems, six imaging modalities, and 21 clinical domains across English and seven major Indian languages.We further propose ArogyaSutra, an actor–critic–based multi-agent framework that combines tool grounding with dual-memory mechanisms to support stepwise, reasoning-aware decision making while explicitly retaining past mistakes to prevent their repeated occurrence.
<p align="left">


<b>Installation</b><br/><br/>

conda create -n asenv python=3.10 -y<br/>
conda activate asenv<br/>
pip install torch==2.6.0 torchvision==0.21.0<br/>
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl<br/>
cd ArogyaSutra<br/>
pip install -e ".[vllm]"<br/>
conda install -c conda-forge compilers<br/><br/>

<b>Tools</b><br/>
conda create -n tools python=3.10 -y<br/>
conda activate tools<br/>
pip install unsloth==2025.9.7<br/>
pip install transformers==4.42.0 fastapi uvicorn matplotlib opencv-python python-multipart<br/>
conda install -c conda-forge compilers<br/>

<br/><br/>
cd tools/Depth-Anything-V2<br/>
mkdir checkpoints<br/>
cd checkpoints<br/>
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth<br/>





<b>1. Start Tools Service:</b><br/>
tmux new -s t_session<br/>
cd ArogyaSutra <br/>
bash tools.sh<br/>

Check depth tool log:  ArogyaSutra/tools/Depth-Anything-V2/Depth.log.<br/>
Check object detection log: ArogyaSutra/tools/LLMDet/Detection.log<br/>

<br/><br/>

<b>2. Generate Actor-Critic training data:</b><br/>
bash generate.sh<br/>



<b>3. Training Qwen2.5:</b><br/>
bash train.sh<br/>

<br/>
<b>4. Strarting VLLM server of the trained model:</b><br/>
bash tserv.sh<br/>
<b>Testing ArogyaSutra:</b><br/>
bash test.sh<br/>



