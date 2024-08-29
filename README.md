# Inpainting-the-Gaps

[arxiv](https://arxiv.org/abs/2406.11534)

## Instructions
1. Clone the repository
```bash
git clone git@github.com:LokeshBadisa/Inpainting-the-Gaps.git
```
2. Install the dependencies
```bash
pip install -r requirements.txt
```
3. Download the dataset from [here](https://github.com/TACJu/PartImageNet?tab=readme-ov-file) and rename it to PartImageNet instead of PartImageNet_OOD.
4. Generate the inpainting dataset
```bash
python3 cdg.py --method inpainting
pip install iopaint
iopaint run --model=migan --device=cuda:0 --image partset32/images --mask=partset32/masks --output=partset32/inpainted
```
5. Download the model weights as specified in `models/ViT/ViT_new.py`
6. Evaluation
```bash
python3 pc.py
python3 dc.py
python3 sd.py
python3 pt.py
```