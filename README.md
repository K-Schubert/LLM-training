# LLM-training
Experimenting with LLMs and training.

## Introduction
Based on the very recent developments in LLMs, I use the methodology of [Self-Instruct](https://arxiv.org/abs/2212.10560) to align the [LLaMA-7B](https://arxiv.org/abs/2302.13971) to replicate findings from the [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) Stanford project. However, instead of using the ```text-davinci-003``` model to generate instruction fine-tuning data, I use the ```gpt-3.5-turbo``` model since it's output is of higher quality and currently competitively priced.

## Instruction and Data Generation
Research has shown that LLMs are not very well aligned with user or human instructions and intents. OpenAI showed with [InstructGPT](https://arxiv.org/abs/2203.02155) that a process of fine-tuning a pre-trained LLM (eg. GPT-3) on high quality curated instruction data helped align models with user intent while rendering them less toxic, hateful, etc. The Self-Instruct paper showed that a supervised fine-tuning approach was already good enough to align a vanilla LLM (eg. GPT-3) with user intent without the need for RLFH and expensive human labeled instructions. Indeed, vanilla LLMs show a great improvement in instruction following using a supervised fine-tuning approach with model-generated synthetic data. As such, I follow the methodology of Self-Instruct in the following way :

- Use the 175 human defined tasks with instructions.
- Generate new instructions based on the human defined tasks using an LLM (LLaMA-7B). Here I follow the paper methodology which defines a batch of 8 new instructions per request and a ratio of 75% human generated instructions to 25% model generated instructions.
- Classify Instructions as either classification/not-classification tasks with a prompt template including a few examples.
- For classification tasks, ask the vanilla LLM to first generate the class label and then the input (eg. for the instruction "Classify the sentiment of the following sentence as either positive, negative or neutral", first generate a class label "positive", then the input sentence "I am feeling good today").
- For not-classification tasks, first generate the input based on the instruction.
- Prompt the vanilla LLM with the {instruction/input} pair and generate an output.
- Use the huggingface training pipeline to fine-tune the vanilla LLM with the synthetic dataset.

## Generated Data Inspection
Filter and clean synthetic data in the following way:

- Remove duplicate instructions and inputs.
- Remove instructions and inputs/outputs with a ROUGE-L overlap over X. ROUGE-L score measures the overlap between the longest common subsequence (LCS, not necessary consecutive words) between two sentences.

## Training Hardware
I use 4 Nvidia Tesla V100 32GB GPUs for training.

## Results and Evaluation

## Next Steps
Training with a larger LLaMA model (eg. LLaMA-13B).

