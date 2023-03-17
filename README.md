# LLM-training
Experimenting with LLMs and training.

## Introduction
Based on the very recent developments in LLMs, I use the methodology of [Self-Instruct](https://arxiv.org/abs/2212.10560) to align the [LLaMA-7B](https://arxiv.org/abs/2302.13971) to replicate findings from the [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) Stanford project. However, instead of using the ```text-davinci-003``` model to generate instruction fine-tuning data, I use the ```gpt-3.5-turbo``` model since it's output is of higher quality and currently competitively priced.

## Instruction Generation
Research has shown that LLMs are not very well aligned with user or human instructions and intents. OpenAI showed with [InstructGPT](https://arxiv.org/abs/2203.02155) that a process of fine-tuning a pre-trained LLM (eg. GPT-3) on high quality curated instruction data helped align models with user intent while rendering them less toxic, hateful, etc. 

## Training Hardware
I use 4 Nvidia Tesla V100 32GB for training.

## Results and Evaluation

## Next Steps
Training with a larger LLaMA model (eg. LLaMA-13B).

