## Setup
First, clone the repository:
```bash
git clone https://github.com/facebookresearch/tome
cd tome
```
Either install the requirements listed above manually, or use our conda environment:
```bash
conda env create --file environment.yml
conda activate tome
```
Then set up the tome package with:
```bash
python setup.py build develop
```
