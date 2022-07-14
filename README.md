# 사랑과 전쟁 얼굴 탐지 및 인식

사랑과 전쟁 얼굴 탐지 및 인식을 위한 학습, 평가, 추론 코드

사랑과 전쟁 데이터는 드라이브에서 업로드

코드 출처 : https://github.com/deepinsight/insightface

## Requirements

기존 arcface 학습용 라이브러리와 동일함

tqdm와 onnxruntime 추가 설치 필요

- Install [PyTorch](http://pytorch.org) (torch>=1.9.0), our doc for [install.md](docs/install.md).
- (Optional) Install [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/), our doc for [install_dali.md](docs/install_dali.md).
- `pip install -r requirement.txt`.

## 실행 준비

폴더를 아래와 같이 정리

드라이브에 업로드된 love_war_dataset과 love_war_inference는 같은 폴더에 위치해야함

## 코드 실행 방법

얼굴 인식 backbone 학습
- python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train.py configs/war_love_r100


얼굴 분류기 학습
- python python train_recognition.py configs/war_love_recognition


얼굴 분류기 성능 평가
- python eval_recognition.py configs/war_love_recognition


얼굴 탐지기와 분류기로 주석 생성
- python inference_face_img.py configs/war_love_recognition
- 주석을 생성하고 싶은 이미지들은 love_war_inference/input에 넣으면 됨
- 출력은 love_war_inference/output에 생성
- 주석 형식은 pascal voc(XML)으로 생성

## 얼굴 인식기 성능

- accuracy
- background 클래스를 포함한 35명의 인물 인식 성능

| total | background | baeksomi | choijeongwon | choisehwan | choiyoungwan |
| ----- | ------ | ----- | ----------- | ---------- | --------- |
| 60.51 | 38.70  | 62.50 | 40.00      | 0.0     | 60.00      |

| hangeurim | hongseongsook | hongyeojin | jangeunbi | jeongnaon | jojeongrae |
| ----- | ------ | ----- | ----------- | ---------- | --------- |
| 71.42 | 0.0  | 40.00 | 68.18      | 60.00     | 60.00      |

| kangjiwoo | kangmoonyoung | kimdeokhyun | kimilran | kimjeongkyun | kimjimin |
| ----- | ------ | ----- | ----------- | ---------- | --------- |
| 68.75 | 100.0  | 66.66 | 0.0     | 66.66    | 86.66      |

| kimseonhyeok | kimsunyoung | kwakhyeonhwa | leejaewook | leejeongsoo | leejihoo |
| ----- | ------ | ----- | ----------- | ---------- | --------- |
| 0.0 | 87.50  | 83.33 | 57.89  | 0.0  | 66.66   |

| leejunwoo | leeseokwoo | leesol | leeyongyi | minjiyoung | moonbobae |
| ----- | ------ | ----- | ----------- | ---------- | --------- |
| 50.00 | 80.00  | 0.0 | 64.86  | 100.0  | 66.66   |

| parkjoohee | parkseonwoo | seokwonsoon | shinsomin | unkiho | yooncheolhyung |
| ----- | ------ | ----- | ----------- | ---------- | --------- |
| 47.05 | 75.00  | 55.55 | 75.00  | 66.66  | 0.0   |

