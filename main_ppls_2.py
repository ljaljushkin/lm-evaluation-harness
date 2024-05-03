from pathlib import Path
import json
import matplotlib.pyplot as plt


tuned_adapters_dir = Path('/home/nlyaly/projects/lm-eval-2/cache/Phi-3-mini-4k-instruct')
DIRS = [
        '10.38_opt_search_q1_wikitext2_loftq_init_R8_Ldugqkvo',
        '10.35_opt_search_q1_wikitext2_loftq_init_R8_Ldug',
        '10.39_opt_search_q1_wikitext2_loftq_init_R8_Ldugqkvo',
]
for exp_dir in DIRS:
    ppls = []
    adapters_dir = tuned_adapters_dir / exp_dir
    x = range(-1, 32)
    xx = []
    for idx in x:
        results_file = adapters_dir / str(idx) / 'results_wikitext.json'
        if results_file.exists():
            with results_file.open('r') as f:
                results = json.load(f)
                word_ppl = results["results"]["wikitext"]["word_perplexity,none"]
                ppls.append(word_ppl)
                xx.append(idx)

    print(ppls)
    plt.grid(axis='both', linestyle='-')
    plt.plot(xx, ppls, **{'marker': 'o'}, label='')
    plt.xticks(xx)
    path = adapters_dir / 'ppls_with_orig.png'
    plt.savefig(path)
    print('saving plot to: ', path)
    plt.clf()
