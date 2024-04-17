# Official source code for LLMSeg: LLM-driven Multimodal Target Volume Contouring in Radiation Oncology


## 1. Environment setting
```
git clone https://github.com/tvseg/MM-LLM-RO.git
pip install -r requirements.txt
```

## 2. Dataset
```
cd ./dataset/external1
download sample dataset from https://1drv.ms/u/s!AhwNodepZ41oi5c2-gC9wn104Db6UQ?e=geDlPs
unzip sample.zip
cd ..
```

## 3. Model checkpoints
```
cd model/llama2
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
cd ..
cd ckpt/multimodal
download model_best.pt from https://1drv.ms/u/s!AhwNodepZ41oi5cpB9lo5U5CbXJz1A?e=tsfaHr
cd ..
```

## 4. Inference
```
WORKDIR=./ckpt/multimodal/model_best.pt
python main.py --pretrained_dir $WORKDIR --context True --n_prompts 2 --context_length 8 --test_mode 2
```
