# MATS Appliction -- Mechanistic Interpretability Research

## Overview
This repository contains the code and research materials related to my application for the MATS program, focusing on mechanistic interpretability of language models using the TransformerLens library. The work is summarized in the `MATS.ipynb` notebook, with supplementary code and data organized in the repository.

## Repository Structure

- **`MATS.ipynb`**: The main Jupyter notebook that documents the experiments and findings related to mechanistic interpretability. It includes analyses on token expressions, module contributions, token representations, and attention head patterns. The notebook also explores gender bias through "Knowledge Neurons" and conducts a preliminary study on "Confidence Regulation Neurons."

- **`dec_llama/`**: Contains scripts for reproducing of linear decomposition for LLaMA model.

- **`ft-training_set/`**: This directory holds data and scripts used for fine-tuning the model on specific tasks, such as reasoning, gender bias analysis, and knowledge neuron identification.

- **`transformer_lens/`**: Includes the TransformerLens library and custom functions utilized in the experiments.

- **`utils/`**: A collection of utility functions and helper scripts to support the analysis and visualization of results within the `MATS.ipynb` notebook.


## Quick Summary

- **Token Expression Analysis**: Investigates how different layers contribute to the model's predictions, focusing on knowledge extraction, IOI prompts, and gender bias.
  
- **Module Contributions**: Analyzes the impact of Attention and MLP modules on predictions through ablation studies.

- **Token Representation Attribution**: Decomposes token contributions within the model to understand how individual tokens influence predictions.

- **In-Context Learning (ICL) Head Analysis**: Identifies effective attention heads in ICL tasks and explores their patterns.

- **Knowledge Neurons**: Explores gender bias through targeted neuron analysis, focusing on the impact of "MAN VECTORS."

- **Confidence Regulation Neurons**: Extends existing research on entropy neurons to examine their role across multiple layers and after fine-tuning.


## Contact
For any questions or discussions, please feel free to reach out via email at haoyanl99@gmail.com.

