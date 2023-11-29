import json
import os
math_data = json.load(open("math_data.json",'r'))

file_path = f'dataset/gsm8k/test.json'
json_data = json.load(open(file_path, 'r'))

new_path = f'dataset/gsm8k/test_20.json'
# if not os.path.exists(new_path):
#     os.mkdir(new_path)

output_data=[]
for i in json_data:
    if i not in math_data:
        output_data.append(i)
        with open(new_path, 'w+') as f:
            json.dump(output_data, f, indent=4)