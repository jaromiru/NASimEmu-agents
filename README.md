This is a repository containing deep RL agents for [NASimEmu](https://github.com/jaromiru/NASimEmu).

## Usage
Install [NASimEmu](https://github.com/jaromiru/NASimEmu) and run as:
```
python main.py <path-to-scenario>
```

## Choosing a model to train / test
Currently, the MLP and invariant models are available. (Un)comment corresponding lines in `nasim_problem/nasim_config.py`. For example:
```
		config.net_class = NASimNetMLP
		# config.net_class = NASimNetInvMAct
```
