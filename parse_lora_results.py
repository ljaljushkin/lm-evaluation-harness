import os
from pathlib import Path
import json

import pandas as pd
cache_dir = Path('cache')

paths_to_result_file = cache_dir.glob('**/log.txt')
paths_to_result_file = sorted(paths_to_result_file, key=os.path.getmtime)
list_exp_dicts = []
for i, path_to_result_file in enumerate(paths_to_result_file):
    # if i > 3:
    #     break
    print(path_to_result_file)
    exp_dir = Path(path_to_result_file).parent
    exp_name = exp_dir.name
    if exp_name.startswith('lora'):
        model_name = exp_dir.parent.name
        with open(path_to_result_file) as f:
            for line in reversed(list(f)):
                if '[ INFO ] [Average]' in line:
                    latency = line.split('Latency: ')[1].split(' ms/token')[0]
                    print(model_name, exp_name, float(latency))
                    exp_dict={
                        'model': model_name,
                        'exp_name': exp_name,
                        'latency (ms/token)': latency,
                    }
                    list_exp_dicts.append(exp_dict)

pd.set_option("display.precision", 2)
df = pd.DataFrame(list_exp_dicts)
print(df)
writer = pd.ExcelWriter('lora_results.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='all', index=False)
wb = writer.book
ws = writer.sheets['all']
wb.close()

