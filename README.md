# **Usage Guidelines**

Please follow the steps below.

## Preparation

We implement Co-DETR using MMDetection V3.3.0 and MMCV V2.1.0. The source code of MMdetection has been included in this repo. We test our models under ```python=3.8.20,pytorch=2.1.0,cuda=12.1```. Other versions may not be compatible.

In a python virtual environment and in the root directory of the project , install packages in the ***requirements.txt*** file.

```shell
pip install -r requirements.txt
```


## Training

Train Co-DINO with 4 GPUs:
```shell
cd mmdetection && sh tools/dist_train.sh my_configs/codino/codino_xxx.py 4 path_to_exp
```

## Test

The test folder contains two subfolders: ***test_data*** and ***test_result***. Among them, ***test_data*** stores the test data, and ***test_result*** stores the test results.

For bar charts:
```shell
python chart_data_extract_bar.py
```

For scatter charts:
```shell
python chart_data_extract_scatter.py
```
You need to modify the path(s) in the file.
