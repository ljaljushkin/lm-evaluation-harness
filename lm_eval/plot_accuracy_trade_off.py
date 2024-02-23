from pathlib import Path
import json
from collections import defaultdict
import os
import matplotlib.pyplot as plt

def plot_tradeoff(x, y, model_name, task_name, log_dir, metric):
    # x=np.array([0,1,2,3,4,5])
    # y=np.array([70,71,72,73,74,75])
    fp32_ref = y[0]
    fig, ax = plt.subplots()
    ax.axhline(fp32_ref, color='red', linestyle='dotted')
    for xx,yy in zip(x,y):
        ax.text(xx, yy + .2, round(yy, 2))
    ax.xaxis.grid()
    # ax.yaxis.grid()
    ax.plot(x,y, marker='o')
    plt.xlabel("Avg. number of pruned experts per layer")
    plt.ylabel(metric)
    plt.title(f'Pruning trade-off for {model_name} on {task_name}')
    path = log_dir / 'tradeoff.png'
    print('Saving plot in: ', path)
    plt.savefig(path)
    plt.close()

metric_per_task = {
    'mrpc': 'acc',
    'sst': 'acc',
    'wikitext': 'word_perplexity',
    'hellaswag': 'acc',
    'gsm8k': 'acc',
    'arc_easy': 'acc',
    'piqa': 'acc',
    'qnli': 'acc',
}

runs_dir=Path('/home/nlyaly/projects/lm-evaluation-harness/results/moe/dfurman__Mixtral-8x7B-Instruct-v0.1')
model_name = runs_dir.name
for task_dir in runs_dir.iterdir():
    if not task_dir.is_dir():
        continue
    task_name = task_dir.name
    metric_name = metric_per_task[task_name]
    for exp_dir in task_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        paths_to_result_file = exp_dir.glob('**/results*.json')
        paths_to_result_file = sorted(paths_to_result_file, key=lambda x: x.name)

        x = []
        y = []
        for i, path_to_result_file in enumerate(paths_to_result_file):
            # print(path_to_result_file)
            with path_to_result_file.open() as f:
                j = json.load(f)
                r = j['results']
                x.append(j['num_experts_to_prune'])
                y.append(r[task_name][metric_name])
        plot_tradeoff(x,y, model_name, task_name, exp_dir, metric_name)


# list_acc_per_task = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# def plot_tradeoff(x, y, model_name, task_name, log_dir, metric):
# for task_name, value in list_acc_per_task.items():
#     for exp_name, results in value.items():
#         x = results['x']
#         y = results['y']
#         log_dir = runs_dir / task_name / exp_name
#         metric_name = metric_per_task[task_name]
#         plot_tradeoff(x,y,model_name, task_name, log_dir, metric_name)
