# ReCoVERR


## Installation

Create conda environment
```
conda create -n vl_rip python=3.9
conda activate vl_rip
```

Install packages:
```
#pip install -e src/modules/LLaVA/
pip install -r requirements.txt

# For using BLIP2/InstructBLIP models
pip install -e src/modules/lavis

python -m spacy download en_core_web_sm
```

Install NLTK resources:
```
import nltk; nltk.download('popular')
```
