# Human Parsing Based Texture Transfer from Single Image to 3D Human via Cross-View Consistency

Code for the paper [Human Parsing Based Texture Transfer from Single Image to 3D Human via Cross-View Consistency (NeurIPS 2020)](https://papers.nips.cc/paper/2020/file/a516a87cfcaef229b342c437fe2b95f7-Paper.pdf). 

## Requirements

- Python 3.6
- Pytorch 1.4

To install Neural Mesh Renderer and Perceptual loss:

```
cd external
sh install_external.sh
```

## Demo

- Download the [trained model](https://drive.google.com/drive/folders/1hbUqIZIOHtpYAnt3yzS_B_wUrpnWVgQ4?usp=sharing).
- Extract and put the model in `cachedir/snapshots`
- Run the demo:

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
sh ./HPBTT/run_eval_market.sh <model_name> <epoch_num> <output_image_path>
```

## Acknowledgement

Our code is based on [cmr](https://github.com/akanazawa/cmr).

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
