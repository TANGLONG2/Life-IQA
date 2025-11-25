# Life-IQA

Checkpoints, logs and source code for paper:'[Life-IQA: Boosting Blind Image Quality Assessment through GCN-enhanced Layer Interaction and MoE-based Feature Decoupling](https://arxiv.org/abs/2511.19024)'


## Dependencies
The code is implemented on Ubuntu20.04 CUDA11.8.
```bash
conda env create -f environment.yml
```

## Usage



#### Swanlab

This project use [Swanlab](https://swanlab.cn/) to log information and report. Remember to adjust the code in main.py to suit your research.

### Training

```bash
./train_test.sh
```

## Citing Life_IQA

If you find this project helpful in your research, please consider citing our papers:

```text
@misc{tang2025lifeiqaboostingblindimage,
      title={Life-IQA: Boosting Blind Image Quality Assessment through GCN-enhanced Layer Interaction and MoE-based Feature Decoupling}, 
      author={Long Tang and Guoquan Zhen and Jie Hao and Jianbo Zhang and Huiyu Duan and Liang Yuan and Guangtao Zhai},
      year={2025},
      eprint={2511.19024},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.19024}, 
}
}
```

## Acknowledgement

We borrowed some parts from the following open-source projects:

* [DEIQT](https://github.com/narthchin/DEIQT)



