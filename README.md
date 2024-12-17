# 6998 Project

(Runze Lin rl3376; Zixuang Fang zf2324)

Our project is based on this paper [Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-intensive Tasks](https://arxiv.org/abs/2305.18395) (NeurIPS 2023).


## Abstract

Knowledge-Augmented Reasoning Distillation (KARD) has emerged as an effective method for transferring knowledge and reasoning capabilities from large language models to smaller, more deployable ones. This paper explores three potential enhancements to the KARD architecture: upgrading the teacher model from GPT-3.5-turbo to GPT-4o, incorporating retrieval-augmented generation (RAG), and implementing a knowledge graph-based retrieval system. We conduct experiments on a subset of 1,000 samples from the StrategyQA dataset, evaluating each enhancement's impact on model performance. Our results demonstrate that utilizing GPT-4o as the teacher model yields the most significant improvement, achieving 54.57% accuracy compared to the baseline of 51.38%. The RAG-enhanced GPT-3.5-turbo shows modest but meaningful improvement, reaching 52.4% accuracy. These findings suggest that teacher model capability currently plays a more crucial role than context augmentation in knowledge distillation effectiveness. Our work provides insights into the relative importance of model capability versus knowledge integration in distillation-based approaches and identifies promising directions for future research in enhancing small language models' performance on knowledge-intensive reasoning tasks.

## Installation
Python version: 3.8.0
```bash
python -m pip install -r requirements.txt
```

## Dataset

Download the raw dataset(MedQA, obQA, StrategyQA) from [this link](https://drive.google.com/file/d/16Niskw2zcvyIdeRUEB2yjU2QQFy2wN3W/view?usp=share_link).

Download the preprocessed data from [this link](https://drive.google.com/file/d/118rvsqpTIHjoOuNgeYmyh7PrlmoKeAm3/view?usp=sharing).

## LM Training Guide
If you want to run Knowledge-Augmented Reasoning Distillation, run the below script:

```bash
sh scripts/run_kard.sh {GPU ID} {Batch Size per GPU} {Model Size:base,large,xl} {Dataset:medqa,strategyqa,obqa}
```

the script for Reasoning Distillation without knowledge augmnetation.

```bash
sh scripts/run_rd.sh {GPU ID} {Batch Size per GPU} {Model Size:base,large,xl} {Dataset:medqa,strategyqa,obqa}
```

Both training script supports multi-gpu training.

For example, if you want to run KARD on the xl-sized LM training on medqa dataset with 4 gpus with batch size 8 per GPU, run as follows:

```bash
sh scripts/run_kard.sh 0,1,2,3 8 xl medqa
```

## Command to modify the data

Better teacher model
```bash
python better_teacher_model.py \
    --file_path path/to/input.json \
    --api_key your-openai-api-key \
    --save_path path/to/output.json
```
3.5 turbo with RAG

```bash
python gpt3.5_RAG.py \
    --file_path path/to/input.json \
    --api_key your-openai-api-key \
    --save_path path/to/output.json
```

Optional arguments:
    --max_retries: Maximum number of retries for failed API calls (default: 3)
    --retry_delay: Delay between retries in seconds (default: 5)
    --temperature: Temperature for GPT-3.5 generation (default: 0.7)

Example:

```bash
python gpt3.5_RAG.py \
    --file_path data/train.json \
    --api_key sk-xxx... \
    --save_path output/improved_train.json
```

## Reranker Training Guide
To train the reranker, check the `reranker` folder.


## Inference Guide
After the LM and reranker training, run the following code:
```bash
python generate_predict.py --checkpoint_path "/path/to/checkpoint/" --retriever_type {sparse,dense} --dataset {medqa_usmle_hf,strategyqa,obqa} --dense_retriever_path "/path/to/retriever/"
```

You can adjust the following hyperparmeters:
- `--max_doc`: adjust the number of maximum passages in the candidate set for reranking.
- `--n_docs`: the number of passages to be appended into the prompt