<div align=center>
<!-- <h1>Avatar: Agent-based Virtual Approach to Large Scale Recommendation Simulation</h1> -->

<h1>Hello Again! LLM-powered Personalized Agent for Long-term Dialogue</h1>



<a href="LICENSE">
<img src="https://img.shields.io/badge/License-MIT-blue" alt="Github license"/>
</a>

<p align="center" style="overflow:hidden;">
 <img src="assets/LD-Agent.png" width="90%" style="margin: -0% -0% -0% -0%;">
</p>

</div>

The official implementation for paper of **"[Hello Again! LLM-powered Personalized Agent for Long-term Dialogue](https://arxiv.org/pdf/2406.05925v1)"**.


<p id="Preparations"></p>  

## ⚙️ Preparations

### Environment Requirements

We recommend the following dependencies:

* Python 3.10.0
* [PyTorch](http://pytorch.org/) 1.13.0
* [Transformers](https://huggingface.co/docs/transformers) (>= 4.32.0)


Then, please install other environment dependencies through:
```bash
pip install -r requirements.txt
```

The recommended GPU memory is more than 32 GB.

### Dataset Preparation

The datasets for event summary, persona extraction, response generation and MSC can be downloaded [here](https://drive.google.com/drive/folders/1ZyYYofzFWW2CxtW0XQZxMQtJ2EtroULX?usp=sharing). Please organize the dataset path as ```LD-Agent/dataset```.


### Metric Preparation

To automatically evaluate response quality, you should download the compressed metric files [here](https://drive.google.com/file/d/1nVDX9Ib796pKXiWKoMSC07Yv04SZ3LuD/view?usp=sharing). Then decompress it and organize it to ```LD-Agent/nlgeval/metric```.

<p id="Quick Start"></p>  

## 💡 Quick Start

### Training

We refer to the training approach of [ChatGLM3-6B](https://github.com/THUDM/ChatGLM3/blob/main/finetune_demo/finetune_hf.py) and separately provide LoRA tuning strategy for event summary, persona extraction, and response generation. You can run the following instructions to train these modules.

**Summarizer**
```bash
bash scripts/summarizer_tuning.sh
```

**Extractor**
```bash
bash scripts/extractor_tuning.sh
```

**Generator**
```bash
bash scripts/generator_tuning.sh
```

You can adjust the detailed training configs in ```Trainer/configs```.

### Evaluation

We provide the evaluation implementations on both [ChatGPT](https://chatgpt.com/) and [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b).

**ChatGPT**

To evaluate using ChatGPT, you can edit the ```${API_KEY}``` in ```scripts/msc_gpt_eval.sh``` to your openai API key and run:
```bash
bash scripts/msc_gpt_eval.sh
```


**ChatGLM**

To evaluate using ChatGLM3-6B, you can run:
```bash
bash scripts/msc_glm_eval.sh
```
Edit the ```${SUMMARIZER}```, ```${EXTRACTOR}```, and ```${GENERATOR}``` to specify the LoRA models used for event summary, persona extraction, and response generation, respectively. The setting of ```"default"``` indicates employing original ChatGLM to the target module.


### Reference

If you found this code useful, please cite the following paper:
```

@article{LD-Agent,
  title={Hello Again! LLM-powered Personalized Agent for Long-term Dialogue},
  author={Li, Hao and Yang, Chenghao and Zhang, An and Deng, Yang and Wang, Xiang and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2406.05925},
  year={2024}
}
```