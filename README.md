### 运行命令

> 无防御手段 + sba Attack + Cora dataset

```python
python main.py --seed=1027 --model=GCN --dataset=Cora --epochs=200 --trigger_size=3 --target_class=0 --attack=sba --vs_ratio=0.005 --defense=none --sba_attack_method=Rand_Gene --sba_trigger_prob=0.5
```

> 无防御手段 + ugba Attack + Cora dataset

```python
python main.py --seed=1027 --model=GCN --dataset=Cora --epochs=200 --trigger_size=3 --target_class=0 --attack=ugba --vs_ratio=0.005 --defense=none --ugba_thrd=0.5 --ugba_trojan_epochs=100 --ugba_inner_epochs=3 --ugba_target_loss_weight=1 --ugba_homo_loss_weight=100 --ugba_homo_boost_thrd=0.8
```

### Cora Dataset

1. EGRAD against SBA attack

```python
python main.py --seed=1027 --model=GCN --dataset=Cora --epochs=200 --trigger_size=3 --target_class=0 --attack=sba --vs_ratio=0.005 --sba_attack_method=Rand_Gene --sba_trigger_prob=0.5 --defense=egrad --egrad_def_epochs=100 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```

2. EGRAD against UGBA attack

```python
python main.py --seed=1027 --model=GCN --dataset=Cora --epochs=200 --trigger_size=3 --target_class=0 --attack=ugba --vs_ratio=0.005 --ugba_thrd=0.5 --ugba_trojan_epochs=100 --ugba_inner_epochs=3 --ugba_target_loss_weight=1 --ugba_homo_loss_weight=100 --ugba_homo_boost_thrd=0.8 --defense=egrad --egrad_def_epochs=100 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```

3. EGRAD against GTA attack

```python
python main.py --seed=1027 --model=GCN --dataset=Cora --epochs=200 --trigger_size=3 --target_class=0 --attack=gta --vs_ratio=0.005 --gta_thrd=0.5 --gta_trojan_epochs=400 --gta_loss_factor=1 --defense=egrad --egrad_def_epochs=200 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```

### Pubmed Dataset

1. EGRAD against SBA attack

```python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --epochs=200 --trigger_size=3 --target_class=0 --attack=sba --vs_ratio=0.005 --sba_attack_method=Rand_Gene --sba_trigger_prob=0.5 --defense=egrad --egrad_def_epochs=1000 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=3 --egrad_recalc_every=5
```

2. EGRAD against UGBA attack

```python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --epochs=200 --trigger_size=3 --target_class=0 --attack=ugba --vs_ratio=0.005 --ugba_thrd=0.5 --ugba_trojan_epochs=100 --ugba_inner_epochs=3 --ugba_target_loss_weight=1 --ugba_homo_loss_weight=100 --ugba_homo_boost_thrd=0.8 --defense=egrad --egrad_def_epochs=1000 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=3 --egrad_recalc_every=5
```
3. EGRAD against GTA attack

```python
python main.py --seed=1027 --model=GCN --dataset=Pubmed --epochs=200 --trigger_size=3 --target_class=0 --attack=gta --vs_ratio=0.005 --gta_thrd=0.5 --gta_trojan_epochs=800 --gta_loss_factor=1 --defense=egrad --egrad_def_epochs=1000 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```

### Flickr Dataset

1. EGRAD against SBA attack

```python
python main.py --seed=1027 --model=GCN --dataset=Flickr --epochs=200 --trigger_size=3 --target_class=0 --attack=sba --vs_ratio=0.005 --sba_attack_method=Rand_Gene --sba_trigger_prob=0.5 --defense=egrad --egrad_def_epochs=100 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```

2. EGRAD against UGBA attack

```python
python main.py --seed=1027 --model=GCN --dataset=Flickr --epochs=200 --trigger_size=3 --target_class=0 --attack=ugba --vs_ratio=0.005 --ugba_thrd=0.5 --ugba_trojan_epochs=400 --ugba_inner_epochs=3 --ugba_target_loss_weight=1 --ugba_homo_loss_weight=100 --ugba_homo_boost_thrd=0.8 --defense=egrad --egrad_def_epochs=100 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```

3. EGRAD against GTA attack

```python
python main.py --seed=1027 --model=GCN --dataset=Flickr --epochs=200 --trigger_size=3 --target_class=0 --attack=gta --vs_ratio=0.005 --gta_thrd=0.5 --gta_trojan_epochs=400 --gta_loss_factor=1 --defense=egrad --egrad_def_epochs=1000 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```

### ogbn-arxiv Dataset

1. EGRAD against SBA attack

```python
python main.py --seed=1027 --model=GCN --dataset=ogbn-arxiv --epochs=200 --trigger_size=3 --target_class=0 --attack=sba --vs_ratio=0.005 --sba_attack_method=Rand_Gene --sba_trigger_prob=0.5 --defense=egrad --egrad_def_epochs=100 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```

2. EGRAD against UGBA attack

```python
python main.py --seed=1027 --model=GCN --dataset=ogbn-arxiv --epochs=200 --trigger_size=3 --target_class=0 --attack=ugba --vs_ratio=0.005 --ugba_thrd=0.5 --ugba_trojan_epochs=400 --ugba_inner_epochs=3 --ugba_target_loss_weight=1 --ugba_homo_loss_weight=100 --ugba_homo_boost_thrd=0.8 --defense=egrad --egrad_def_epochs=100 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```

3. EGRAD against GTA attack

```python
python main.py --seed=1027 --model=GCN --dataset=ogbn-arxiv --epochs=200 --trigger_size=3 --target_class=0 --attack=gta --vs_ratio=0.005 --gta_thrd=0.5 --gta_trojan_epochs=400 --gta_loss_factor=1 --defense=egrad --egrad_def_epochs=1000 --egrad_alpha=0.5 --egrad_eta=0.9 --egrad_tau=0.5 --egrad_gamma=0.1 --egrad_lambda1=0.5 --egrad_R_scale=4 --egrad_recalc_every=10
```