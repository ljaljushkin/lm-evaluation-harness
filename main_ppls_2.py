from pathlib import Path
import json
import matplotlib.pyplot as plt


tuned_adapters_dir = Path('/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-tuned-alpha-7b')
DIRS = [
    '25.44_opt_search_loftq_ptb_synth_faster_R8_Ldd_VALIDATE',
    '25.58_opt_search_loftq_ptb_synth_R8_Ldd_VALIDATE',
    '25.98_opt_search_loftq_ptb_max_len_small_steps_R8_Ldd_VALIDATE',
]
for exp_dir in DIRS:
    ppls = []
    adapters_dir = tuned_adapters_dir / exp_dir
    x = range(-1, 16)
    xx = []
    for idx in x:
        results_file = adapters_dir / str(idx) / 'results_wikitext.json'
        if results_file.exists():
            with results_file.open('r') as f:
                results = json.load(f)
                word_ppl = results["results"]["wikitext"]["word_perplexity"]
                ppls.append(word_ppl)
                xx.append(idx)

    print(ppls)
    plt.grid(axis='both', linestyle='-')
    plt.plot(xx, ppls, **{'marker': 'o'}, label='')
    plt.xticks(xx)
    plt.savefig(adapters_dir / 'ppls.png')
    print('saving plot to: ', adapters_dir / 'ppls.png')
    plt.clf()
