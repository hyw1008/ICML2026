Code for Respecting Modality Gap in Post-hoc Out-of-distribution Detection with Pre-trained Vision-Language Models.

Run:

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
bash scripts/run_main_imagenet_ood.sh --data_root /path/to/data
bash scripts/run_openood.sh cifar10 --data_root /path/to/data
bash scripts/run_openood.sh cifar100 --data_root /path/to/data
```
