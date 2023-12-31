from concurrent.futures import ProcessPoolExecutor
import queue
import subprocess
import json
import time
import os
def evaluate(dataset, gpu):
    print('*******dataset:', dataset)
    start = time.time()
    command = f"CUDA_VISIBLE_DEVICES={gpu} python evaluate.py \
               --model LLaMA-7B \
               --adapter LoRA \
               --dataset {dataset} \
               --base_model 'yahma/llama-7b-hf' \
               --lora_weights './trained_models/Lora4'"
    result = subprocess.run(command, shell=True, text=True, capture_output=False)
    end = time.time()
    print(str(end-start))
    print(f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}")
    save_file = f'multi_experiment/21/{dataset}.json'
    with open(save_file, 'w+') as f:
        json.dump(f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}", f, indent=4)
        json.dump(str(end-start), f, indent=4)
    return gpu

dir_path= 'multi_experiment/21/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

datasets = ['AQuA','AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'SVAMP'] # 删掉了SVAMP 'SingleEq', , 'gsm8k' 'AQuA','AddSub', 'MultiArith', 'SingleEq',
# datasets = ['SVAMP', 'AQuA']
gpus = [0]
tasks_queue = queue.Queue()
gpu_queue = queue.Queue()

for gpu in gpus:
    gpu_queue.put(gpu)
for task in datasets:
    tasks_queue.put(task)

num_processes = min(len(datasets), len(gpus))  # number of processes to run in parallel

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = [executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()) for i in range(num_processes)]
    for future in futures:
        gpu_id = future.result()
        gpu_queue.put(gpu_id)
        if tasks_queue.qsize() > 0:
            futures.append(executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()))





