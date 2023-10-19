<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


<h3 align="center">
    <t>HLPT: Hybrid LoRA-Prefix Tuning </t>
</h3>
The method optimally combines both LoRA and Prefix-Tuning to address the varying needs of different layers. HLPT offers performance improvements while being efficient, with results indicating faster model convergence and reduced training resources. This approach demonstrates robustness and general applicability across different datasets and models, showing potential for training efficiency and parameter reduction in large language models.



## Setup

Install dependencies
```bash
pip install -r requirements.txt
cd peft/
pip install -e .
```

Due to the modifications required by our method, please replace the "modeling_llama.py" file under '/transformers/models/llama' and the "modeling_gptj.py" file under '/transformers/models/gptj' with the files we provide.

You can directly use the HLPT method with the code we provide.To use H2LPT method,You can modify the code by following the comments in "peft/src/peft/tuners/lora.py" and "modeling_llama.py/modeling_gptj.py"
## Training

This file contains some code related to prompt construction and tokenization.In this file, specify different adapters and different sets of data, so that different models can be trained. 

You can use 'bash finetune.sh' to train the model with HLPT method

## Evaluation

To evaluate the performance of the finetuned model on the six Arithmetic Reasoning tasks, you can use 'python multi_dataset_eval.py':

## Acknowledgement

This repo largely benefits from LLM-Adapters(https://github.com/AGI-Edgerunners/LLM-Adapters). Thanks for the wonderful work. 