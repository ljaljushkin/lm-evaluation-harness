from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
import torch
import torch.nn.functional as F
from functools import partial
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')

RESULTS_ROOT = Path('results/moe')
class Collector:
    def __init__(self, name, total_num_experts, top_k, device):
        self.name = name
        self.total_num_experts = total_num_experts
        self.top_k = top_k
        self.alpha_score = torch.zeros([total_num_experts], dtype=torch.float64).to(device)
        self.hit_rate = torch.zeros([total_num_experts], dtype=torch.float64).to(device)
        self.num_adds = 0
        self.device = device

    def add(self, alpha_score, hit_rate):
        self.alpha_score += alpha_score
        self.hit_rate += hit_rate
        self.num_adds += 1

    def get_avg_score(self):
        if self.num_adds == 0:
            raise ValueError('no stats collected')
        return self.alpha_score / self.num_adds

    def get_avg_rate(self):
        if self.num_adds == 0:
            raise ValueError('no stats collected')
        return self.hit_rate / self.num_adds

class MoEPruner:
    def __init__(self, model, task_name, is_prune, prune_metric, model_name, prune_strategy='GLOBAL_THRESHOLD', ratio=6/8):
        self.model = model
        self.task_name = task_name
        self.collectors = []
        self.hook_handles = []
        self.is_prune = is_prune
        self.prune_metric = prune_metric
        model_name = model_name.replace('/', '__')
        self.results_dir = RESULTS_ROOT / model_name / self.task_name
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.scores_path = self.results_dir / 'score_per_layer.scv'
        self.rates_path = self.results_dir / 'rate_per_layer.scv'
        self.metric_path = self.scores_path if self.prune_metric == 'scores' else self.rates_path
        mode_str = 'pruning' if self.is_prune else 'calibration'
        self.log_path = self.results_dir / (mode_str + '_log.txt')
        self.prune_strategy = prune_strategy
        self.ratio = ratio
        print(f'Pruner is working in the {mode_str} mode')
        print('Log file: ', self.log_path.resolve())



    def __exit__(self, *args, **kwargs):
        if self.is_prune:
            return
        self._save_results()

    def __enter__(self):
        with self.log_path.open('w') as f, redirect_stdout(f), redirect_stderr(f):
            if self.is_prune:
                self._prune_experts()
            else:
                self._setup_hooks()

    def _setup_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, MixtralSparseMoeBlock):
                device = next(module.parameters()).device
                collector = Collector(name, module.num_experts, module.top_k, device)
                self.hook_handles.append(module.gate.register_forward_hook(partial(MoEPruner.gate_spy, collector=collector)))
                self.collectors.append(collector)

    def _save_results(self):
        with self.log_path.open('w') as f, redirect_stdout(f), redirect_stderr(f):
            print('saving results to: ', str(self.results_dir))
            score_per_layer = torch.stack([collector.get_avg_score().to('cpu') for collector in self.collectors])
            rate_per_layer = torch.stack([collector.get_avg_rate().to('cpu') for collector in self.collectors])
            print('total hit rate: ', rate_per_layer.mean(dim=0))
            print('total alpha score: ', score_per_layer.mean(dim=0))

            score_per_layer = score_per_layer.cpu()
            rate_per_layer = rate_per_layer.cpu()
            pd.DataFrame(score_per_layer).to_csv(self.scores_path)
            pd.DataFrame(rate_per_layer).to_csv(self.rates_path)

            pd.DataFrame(score_per_layer).plot(title='Alpha score on ' + self.task_name)
            plt.xlabel("Expert ID")
            plt.ylabel("Metric")
            plt.savefig(self.scores_path.with_suffix('.png'))

        pd.DataFrame(rate_per_layer).plot(title='Hit rate on ' + self.task_name)
        plt.xlabel("Expert ID")
        plt.ylabel("Metric")
        plt.savefig(self.rates_path.with_suffix('.png'))

        for handle in self.hook_handles:
            handle.remove()

    @staticmethod
    def gate_prune(module, input_, output, pruning_mask):
        # pruning_mask: [NumExperts]
        router_logits = output # [SeqLen, NumExperts]
        min_value = torch.finfo(output.dtype).min
        pruned_expert_indices = pruning_mask == 0
        min_value = torch.finfo(router_logits.dtype).min
        router_logits[:, pruned_expert_indices] = min_value
        # print('router_logits', router_logits[0,:]) # [SeqLen, NumExperts]
        # all_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # [SeqLen, NumExperts]
        # print('all_routing_weights', all_routing_weights[0,:]) # [SeqLen, NumExperts]
        return router_logits

    @staticmethod
    def gate_spy(module, input_, output, collector):
        router_logits = output
        # print('router_logits', router_logits[0,:])
        top_k = collector.top_k
        all_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # [SeqLen, NumExperts]
        # print('all_routing_weights', all_routing_weights[0,:])
        _, selected_experts = torch.topk(all_routing_weights, top_k, dim=-1) # [SeqLen, top_k]
        # print('selected_experts', selected_experts[0,:])
        num_tokens = selected_experts.shape[0] # SeqLen
        binary_mask = torch.zeros([num_tokens, collector.total_num_experts], dtype=torch.long).to(collector.device) # [SeqLen, NumExperts]
        binary_mask.scatter_(1, selected_experts, 1) # [SeqLen, NumExperts]

        selected_weights = binary_mask * all_routing_weights
        alpha_score_for_one_sequence = selected_weights.mean(dim=0) # [NumExperts]
        hit_rate_for_one_sequence = binary_mask.sum(dim=0) / num_tokens / top_k # [NumExperts]
        collector.add(alpha_score_for_one_sequence, hit_rate_for_one_sequence)
        sep = '\n\t\t'
        # print(f"{name}{sep}top_k={top_k}{sep}routing_weights={routing_weights}{sep}selected_experts={selected_experts}{sep}hit_rate={hit_rate_for_one_sequence}{sep}alpha_score={alpha_score_for_one_sequence}{sep}")
        # print(f"{collector.name}{sep}top_k={top_k}{sep}hit_rate={hit_rate_for_one_sequence}{sep}alpha_score={alpha_score_for_one_sequence}{sep}")

    def _get_pruning_masks(self):
        df = pd.read_csv(self.metric_path)
        scores = torch.tensor(df.iloc[:,1:].values)

        if self.prune_strategy == 'MIN_ON_LAYER':
            min_expert_id = scores.min(dim=1)[1]
            # TODO: hardcoded num exports
            pruning_masks = abs(1 - torch.nn.functional.one_hot(min_expert_id, num_classes=8))
        elif self.prune_strategy == 'GLOBAL_THRESHOLD':
            num_scores = scores.numel()
            border_idx = int(num_scores * self.ratio)
            threshold = scores.reshape([-1, 1]).sort(dim=0)[0][border_idx]
            # Specifically uint8 mask for Layer4Bit gate
            pruning_masks = torch.where(scores>=threshold, torch.ByteTensor([1]), torch.ByteTensor([0]))
            # pruning_masks = torch.where(scores>=threshold, 1, 0)

            fig, ax = plt.subplots(label='Number of active experts per layer')
            ax.plot(pruning_masks.sum(dim=1))
            # TODO: plot bars
            ax.plot(torch.ones_like(pruning_masks).sum(dim=1), color='red', linestyle='dotted')
            plt.xlabel("Layer ID")
            plt.ylabel("Num active experts")
            plt.savefig(self.metric_path.parent / f'active_experts_r{self.ratio:.2f}.png')

        return pruning_masks

    def _prune_experts(self):
        pruning_masks = self._get_pruning_masks()
        # device = next(self.model.parameters()).device
        # pruning_masks = pruning_masks.to(device)
        print(pruning_masks)
        i = 0
        for name, module in self.model.named_modules():
            if isinstance(module, MixtralSparseMoeBlock):
                print('name=', name, '\n\t\tpruning_masks=', pruning_masks[i]) # [NumExperts]
                self.hook_handles.append(module.gate.register_forward_hook(partial(MoEPruner.gate_prune, pruning_mask=pruning_masks[i])))
                i+=1

        # for name, module in self.model.named_modules():
        #     if isinstance(module, MixtralSparseMoeBlock):
        #         with torch.no_grad():
        #             # TODO: check that internal dimension and total number of experts are even!! check in_features, out_features
        #             hidden_dim = module.gate.in_features
        #             num_experts = module.gate.out_features

        #             assert hidden_dim % 2 == 0, f'hidden_dim={hidden_dim} is not even'
        #             assert num_experts % 2 == 0, f'num_experts={num_experts} is not even'

        #             mask = torch.unsqueeze(pruning_masks[i], dim=1)
        #             print(module.gate.weight.shape)
        #             print(mask.shape)
        #             device = module.gate.weight.device
        #             print(device)
        #             mask = mask.to(device)
        #             print(mask)
        #             original_shape = module.gate.weight.shape
        #             module.gate.weight.data = (module.gate.weight.reshape(num_experts, hidden_dim // 2) * mask).reshape(original_shape)

        #             # TODO: not optimal
        #             # TODO: override with less number of experts, update matmul params accordingly
        #             # w = module.gate.weight.t() * pruning_masks[i]
        #             # module.gate.weight.data = w.t().data
        #             print(name, '\n\t', module.gate.weight.reshape(num_experts, hidden_dim // 2)[:,0])
        #             i += 1