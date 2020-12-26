# Human Parsing Based Texture Transfer from Single Image to 3D Human via Cross-View Consistency

Code for the paper [Human Parsing Based Texture Transfer from Single Image to 3D Human via Cross-View Consistency (NeurIPS 2020)](https://papers.nips.cc/paper/2020/file/a516a87cfcaef229b342c437fe2b95f7-Paper.pdf). 

## Requirements

- Cuda 9.2
- Python 3.6
- Pytorch 1.4
- Opencv 3.4.2

Other requirements:

```
pip install -r requirements.txt
```

To install Neural Mesh Renderer and Perceptual loss:

```
cd ./HPBTT/external
sh install_external.sh
```

## Datasets

- [Market-1501](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- [DeepFashion (In-shop Clothes Retrieval Benchmark)](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

## Demo

- Download the [trained models](https://drive.google.com/drive/folders/1zTxalRr6RnJ-lK3_u2JB23ejF1yBK1Ao?usp=sharing).
- Extract and put the models of Market-1501 and DeepFashion in `./HPBTT/cachedir/snapshots` and HMR models in `./HPBTT/external/hmr`
- Run the demo:

```
sh ./HPBTT/run_demo_market.sh <model_name> <epoch_num> <input_image_name>
```

## Training

- Download the [Market-1501 dataset](https://drive.google.com/file/d/1HGRDRrYr0nLhQzQmAFc2Odws_bn3q9Jo/view?usp=sharing).
- Extract and put the dataset in `./dataset`
- Generate input images and their SMPL parameters:

```
python -m HPBTT.experiments.hmr_market
```

- Train the model:

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
