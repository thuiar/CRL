# CRL
Implementation of the research paper [Consistent Representation Learning for Continual Relation Extraction](https://arxiv.org/abs/2203.02721) (Findings of ACL 2022)

## Framwork of consistent representation learning

Memory Replay Architecture

<div align="center">
<img src=figs/crl.png width=80% />
</div>

## Dependencies

Use anaconda to create python environment:

> conda create --name yourname python=3.8 \
> conda activate yourname

Install Pytorch (suggestions>=1.7) and related environmental dependencies:

> pip install -r requirements.txt

Pre-trained BERT weights:
* Download *bert-base-uncased* into the *datasets/* directory [[google drive]](https://drive.google.com/drive/folders/1BGNdXrxy6W_sWaI9DasykTj36sMOoOGK).

### Run the Code

> python run_continual.py --dataname FewRel

## Citation
Please cite our paper if you find our work useful for your research:

```
@misc{zhao2022consistent,
      title={Consistent Representation Learning for Continual Relation Extraction}, 
      author={Kang Zhao and Hua Xu and Jiangong Yang and Kai Gao},
      year={2022},
      eprint={2203.02721},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

