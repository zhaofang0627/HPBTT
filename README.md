# Human Parsing Based Texture Transfer from Single Image to 3D Human via Cross-View Consistency

Code for the paper [Human Parsing Based Texture Transfer from Single Image to 3D Human via Cross-View Consistency (NeurIPS 2020)](https://papers.nips.cc/paper/2020/file/a516a87cfcaef229b342c437fe2b95f7-Paper.pdf). 

## Requirements

- Python 3.6
- Pytorch 1.4

To install Neural Mesh Renderer and Perceptual loss:

```
cd external
bash install_external.sh
```

## Demo

To run the demo:

```
sh ./HPBTT/run_demo_market.sh <model_name> <epoch_num> <input_image_name>
```

## Training

To train the model:

```
python -m HPBTT.experiments.train_market --name <model_name>
```

## Evaluation

To evaluate the model:

```
python -m HPBTT.eval_market
python -m HPBTT.eval_market --name <model_name> --num_train_epoch <epoch_num> --nohmr --img_path <pred_image_path>
python -m cmr_py3.ssim_score_market <pred_image_path>
```

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{zhao2020human,
  title={Human Parsing Based Texture Transfer from Single Image to 3D Human via Cross-View Consistency},
  author={Zhao, Fang and Liao, Shengcai and Zhang, Kaihao and Shao, Ling},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
