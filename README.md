# SatCLA
The code demo of SatCLA.
## Installation
We have tested SatCLA based on Python 3.11 on Ubuntu 20.04, theoretically it should also work on other operating systems. To get the dependencies of SatCLA , it is sufficient to run the following command. 
`pip install -r requirements.txt`

## The structure of the repository
This directory includes implementations for evaluating Automatic Annotation Performance, assessing the performance of LLMs enhanced with structured prompts, and Collaborative Annotation Efficiency.
Prerequisites for running the code:
1. PyTorch and related dependency packages need to be installed.
2. LLMs can be deployed locally in advance using Ollama.
3. Collaborative annotation involves multiple nodes, so multiple computing devices are required to handle collaborative annotation tasks.

## Datasets and Vector Database
</br>We integrated eight publicly available remote sensing datasets—to build a comprehensive [vector feature database](https://www.hostize.com/zh/v/KuidvWj_tS) covering 45 land-cover categories. Here, we provide the source files for this vector feature database to reproduce the experimental results. This is an essential part of the execution process and can be placed in the '../vector_db/annoy_index ' directory.
</br>In addition, we also provided the raw data of the three datasets used in the paper, including [UC Merced Land-Use, RSI-CB256 and RSSCN7](https://www.hostize.com/zh/v/dCuK7_eqlb). After downloading the dataset, you can place it in the `../query_data` directory to use it.

## Program Execution Workflow
</br>1.The prior expert knowledge and vector feature database have been generated in this step, and the corresponding files have already been uploaded to the directory.
</br>2.Perform the initial round of annotation using the LLMs on the master node. `python generate_label_ucm.py`
</br>3.Perform the re-annotation process on the collaborative nodes. Note that if you do not have multiple collaborative nodes, you can perform collaborative annotation using virtual nodes.`python re_tag.py`
</br>4.Evaluate the quality of the finally generated label.`python evaluate_label.py`

## Explanation of the Efficiency Experiment in Annotation Task Processing.
Due to the differences in computational power among devices, the processing efficiency and latency can vary significantly when performing annotation tasks. These variations may affect the overall performance and speed of collaborative annotation.
If you aim to reproduce experimental results with collaborative annotation efficiency, it is recommended to:
- Configure the same computing environment (e.g., CPU/GPU specifications, memory capacity),
- Ensure identical network bandwidth settings across all nodes,
- Follow the setup details provided in the paper.
</br>This ensures a fair comparison and accurate evaluation of the annotation efficiency under controlled conditions.



