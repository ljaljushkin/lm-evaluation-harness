import os
from pathlib import Path
import json

import pandas as pd
runs_dir = Path('runs')

# {
#   "results": {
#     "lambada_openai": {
#       "ppl": 7.069002651432085,
#       "ppl_stderr": 0.2724252023704873,
#       "acc": 0.6202212303512517,
#       "acc_stderr": 0.0067616195103771684
#     },
#     # 'fp32_ref': {
#         # "ppl":
#         # "acc":
#     # }
#   },
#   "config": {
#     "model_args": "pretrained=/home/devuser/nlyalyus/projects/lm-evaluation-harness/cache/open_llama_3b/nf4_ov",
#     "limit": null,
#   },
#   "time": {
#     "eval": 1663.7129156589508
#   },
#   "experiment_config": {
#     "model_id": "openlm-research/open_llama_3b",
#     "group_size": 64,
#     "mode": "nf4",
#     "is_mixed": false,
#     "is_fp32": false,
#     "exp_name": "nf4_ov",
#   }
# }

FP32_REFS = {
    'opt-125m': (37.86, 25.99),
    'dolly-v2-3b': (62.97, 5.014),
    'open_llama_3b': (65.418, 6.255),
    'open_llama_13b': (0,0),
    'opt-6.7b': (67.688, 4.253),
    'bloom-7b1': (57.636, 6.619),
    'RedPajama-INCITE-7B-Instruct': (68.950, 4.153),
    'dolly-v2-12b': (64.311, 4.798),
    'Llama-2-7b-chat-hf': (70.58, 3.278),
    'Llama-2-13b-chat-hf': (73.122, 2.916),
    # 'chatglm2-6b',
}

paths_to_result_file = runs_dir.glob('**/results.json')
paths_to_result_file = sorted(paths_to_result_file, key=os.path.getmtime)
list_exp_dicts = []
for i, path_to_result_file in enumerate(paths_to_result_file):
    # if i > 3:
    #     break
    print(path_to_result_file)
    folder_with_date = str(Path(path_to_result_file).parent.name)
    with path_to_result_file.open() as f:
        j = json.load(f)
        r = j['results']
        model_size = j.get('model_size', 0)
        ov_version = j.get('ov_version', 0)
        c = j['config']
        limit = c['limit']
        model_args = c['model_args']
        exp_name = Path(model_args).name
        model_name = Path(model_args).parent.name
        # TODO: get date and sort by date
        day, time = folder_with_date.split('_')[-2:]
        # print(day, time)
        if True: #day == 'Sep28':
            exp_dict={
                'model': model_name,
                'mode': exp_name,
                'date': f'{day}_{time}',
                'limit': limit,
            }
            for task_name, rr in r.items():
                if task_name == 'lambada_openai':
                    exp_dict['acc'] = rr['acc'] * 100
                    exp_dict['ppl'] = rr['ppl']

                    ref_acc, ref_ppl = FP32_REFS[model_name]
                    exp_dict['diff_acc'] = exp_dict['acc'] - ref_acc
                    exp_dict['diff_ppl'] = exp_dict['ppl'] - ref_ppl
            exp_dict['model_size'] = model_size
            exp_dict['ov_version'] = ov_version
            list_exp_dicts.append(exp_dict)
            print(json.dumps(exp_dict, indent=4))
pd.set_option("display.precision", 2)
df = pd.DataFrame(list_exp_dicts)
df.sort_values(by=['date'], inplace=True)
# df.style.applymap(color, subset=['diff_acc'])
# df.style.highlight_between(left=-1, right=0, props='background-color:green;')

print(df)
# df.to_excel("results.xlsx")
writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='all', index=False)
(max_row, max_col) = df.shape
wb = writer.book
green_format = wb.add_format({'bg_color':'#9BBB59'})
ws = writer.sheets['all']
ws.conditional_format(f'G2:G{max_row}' , {'type': 'cell', 'criteria': '>=', 'value': -1, 'format':  green_format})
wb.close()

