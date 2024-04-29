
<h1 align="center"> nach0  </h1>
<h3 align="center"> Multimodal Natural and Chemical Languages Foundation Model </h3>
<p align="center">
  📃 <a href="https://arxiv.org/abs/2311.12410" target="_blank">Paper</a> • ⏬ <a href="https://huggingface.co/datasets/insilicomedicine/nach0" target="_blank">Base nach0</a> • ⏬ <a href="https://huggingface.co/datasets/insilicomedicine/nach0" target="_blank">Large nach0</a> <br>
</p>
<div align=center><img src="images/nach0_Pub_2.png" width="70%" height="70%" /></div>
<h2 id="1">Overview</h2>

- nach0 is a multi-domain and multi-task encoder-decoder LLM pre-trained on unlabeled text from scientific literature, patents, and molecule strings to incorporate a range of chemical and linguistic knowledge.

- We employed instruction tuning, where specific task-related instructions are utilized to fine-tune nach0 for the final set of tasks. To train nach0 effectively, we leverage the NeMo framework, enabling efficient parallel optimization of both base and large model versions. 

- Extensive experiments demonstrate that our model outperforms state-of-the-art baselines on single-domain and cross-domain tasks. Furthermore, it can generate high-quality outputs in molecular and textual formats, showcasing its effectiveness in multi-domain setups.

<h2 id="1">Tasks</h2>
Datasets used for training and evaluation. Colour represents the type of tasks. Yellow and blue datasets are single-domain, typically requiring regression/classification losses or generation in the target domain (natural language or SMILES strings). Gradients from yellow to blue represent cross-domain generation tasks that require natural language input and SMILES output, or vise versa.
<div align=center><img src="images/nach0_Pub_1.png" width="70%" height="70%" /></div>

<h2> Model Usage Guide</h2>


<h3> References</h3>
If you use our repository, please cite the following related paper:

```
@inproceedings{....
}
```
