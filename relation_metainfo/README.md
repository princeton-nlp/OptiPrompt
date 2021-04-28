This folder contains (discrete-token-based) prompts used in previous factual probing works. These files can be used to reproduce their probing results on LAMA.

Prompts:
* `LAMA_relations.jsonl`: manually-defined prompts used in [LAMA](https://github.com/facebookresearch/LAMA). We use these prompts to initialize the dense vectors in OptiPrompt.
* `LPAQA_relations.jsonl`: the top-1 prompts used in [LPAQA](https://github.com/jzbjyb/LPAQA).
* `AutoPrompt_relations`: the prompts used in [AutoPrompt](https://github.com/ucinlp/autoprompt). They are optimized on the same training set with our OptiPrompt.