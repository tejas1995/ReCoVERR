# ReCoVERR

This is the code for the paper [Selective "Selective Prediction": Reducing Unnecessary Abstention in Vision-Language Reasoning](https://arxiv.org/abs/2402.15610), published in ACL Findings 2024.
## Installation

Create conda environment
```
conda create -n recoverr python=3.10
conda activate recoverr
```

Install packages:
```
#pip install -e src/modules/LLaVA/
pip install -r requirements.txt

# For using BLIP2/InstructBLIP models
pip install -e src/modules/lavis

# For using LLaVA models
pip install -e transformers --upgrade

python -m spacy download en_core_web_sm
```

Install NLTK resources:
```
import nltk; nltk.download('popular')
```

## Running ReCoVERR for your VLM

All relevant files can be found in the `src/` directory.

### Preliminaries: OpenAI and WandB Config Files

Before running ReCoVERR, you will need to create a configuration file with your OpenAI API key and org key. The file should be structured as
```
api_key: XXX
org_key: XXX
```

Modify line 42 of `utils/openai_utils.py` to point to your OpenAI config file.

You will also need to create a configuration file with your WandB API key. The file should be structured as
```
entity: WANDB_USERNAME
api_key: WANDB_API_KEY
project_name: XXX
```

You can pass the path to this file as an argument to the `--wandb_config_file` argument when running the scripts. Alternatively, you can modify L41 and L433 of `direct_vqa.py` and `run_recoverr.py` to point to your WandB config file, so you dont have to pass the path as an argument each time.

### Evaluating Vanilla Selective Prediction

To establish the vanilla selective prediction baselines, you will need to extract the VLM's predictions on a calibration set, as well as an evaluation set. Optionally, you also need a model to train the confidence calibrator. In the case of A-OKVQA, we used the validation set as our evaluation set, 5,000 training examples from the training data as our calibration set, and the remaining training examples as our calibration training set.

You will need to create a config file for your VLM in `configs/vlm_configs/`. To extract the VLM's predictions, run the `direct_vqa.py` script as follows:
```
python -m direct_vqa --vlm_config_file ${VLM_CONFIG_FILE} \
    --dataset ${DATASET} \
    --split ${SPLIT} \
    --num_examples ${NUM_EXAMPLES}
```

After extracting the VLM's predictions for the calibration and evaluation sets, you can use the `Vanilla Selective Prediction.ipynb` notebook to evaluate the vanilla selective prediction baseline. This notebook should, for your VLM and a specified risk level $r$, give you the corresponding confidence threshold $\gamma_{@r}$, as well as the coverage of Vanilla Selective Prediction at that risk level.

### Running ReCoVERR

Create a config file in `configs/recoverr_configs/` (such as [this one](https://github.com/tejas1995/ReCoVERR/blob/main/src/configs/recoverr_configs/aokvqa/blip2ft5xl_uncalibrated_vlm/chatgpt_qgen-flant5xl_llm-lvis_objdet.yaml)).

To run ReCoVERR, use the `run_recoverr.py` script as follows:
```
python -m run_recoverr --config_file ${RECOVERR_CONFIG_FILE} \
    --dataset ${AOKVQA} \
    --split ${SPLIT} \
    --num_examples ${NUM_EXAMPLES}
```

## Citation

If you use this codebase, please cite the following paper:

```
@inproceedings{srinivasan2024selective,
  title={Selective" Selective Prediction": Reducing Unnecessary Abstention in Vision-Language Reasoning},
  author={Srinivasan, Tejas and Hessel, Jack and Gupta, Tanmay and Lin, Bill Yuchen and Choi, Yejin and Thomason, Jesse and Chandu, Khyathi Raghavi},
  booktitle={Findings of the Association for Computational Linguistics (ACL)},
  year={2024}
}
```
