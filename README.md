# textrnn-pytorch

## nvidia-docker
nvidia-docker run -v /home/clay/workspace:/workspace --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/tensorflow:19.05-py3



## training
python train.py --mode train --file ./training/metal_song_titles/source/The-Collected-Works-of-HP-Lovecraft_djvu_poems_clean.txt --session metal03 --number 4000

## prediction
python train.py --mode predict --session metal03

python train.py --mode predict --file ./training/metal_song_titles/source/The-Collected-Works-of-HP-Lovecraft_djvu_poems_clean.txt --session metal04 --initial "I am very busy"


__Python VERSION: 3.7.7 (default, May  7 2020, 21:25:33) 
[GCC 7.3.0]
__pyTorch VERSION: 1.3.1
__CUDA AVILABLE: True
__CUDNN VERSION: 7605
__Number CUDA Devices: 1
__Device: cuda GeForce RTX 2080 Ti
Active CUDA Device: GPU 0
Available devices  1
Current cuda device  0
Vocabulary size 5778
High This Hybrid Valiant Tho' Deeds Flutter


## running the title app
FLASK_APP=app.py APP_CONFIG=textrnn.cfg flask run --host=0.0.0.0 --port=5001 

## curl to get text from the the tile app
curl http://localhost:5001/title?session_id=metal04

{"title":"Glimpse No Wind Swear"}


## training using the NVIDIA docker container
docker run --gpus all --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 -it --rm \
    -v $(pwd)/workspace:/workspace nvcr.io/nvidia/pytorch:20.06-py3 \
    python train.py --mode train \
    --file /workspace/training/hplc/metal_gothic_poetry.txt \
    --session metal03 --number 4000 

## building the docker container
`docker build . -t textrnn:latest`

### train
```docker run --gpus all --shm-size=1g --ulimit memlock=-1     --ulimit stack=67108864 -it --rm     -v $(pwd)/workspace:/workspace textrnn:latest python src/train.py --mode train  --file /workspace/training/hplc/metal_gothic_poetry.txt  --session metal05 --number 1000 ```

docker run --gpus all --shm-size=1g --ulimit memlock=-1     --ulimit stack=67108864 -it --rm     -v $(pwd)/workspace:/workspace textrnn:latest python src/train.py --mode train  --file /workspace/training/hplc/metal_gothic_poetry.txt  --session reflect_01 --number 40000


### predict
```docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm  -v $(pwd)/workspace:/workspace textrnn:latest python src/train.py --mode predict --session metal05```

and

```docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm  -v $(pwd)/workspace:/workspace textrnn:latest python src/predict.py --session metal05 --predict 500 --lines 10```

and 

```
docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm  -v $(pwd)/workspace:/workspace textrnn:latest python src/predict.py --session metal05 --predict 500 --lines 10
```

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm  -v $(pwd)/workspace:/workspace textrnn:latest python src/predict.py --session metal05 --predic 500 --lines 10 --initial "disquieting earth" -o /workspace/reflect/mar_3_2021


docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm  -v $(pwd)/workspace:/workspace textrnn:latest python src/generate.py --session metal05 --predic 800 --lines 20 --initial "disquieting earth"

python src/generate.py --session reflect_02 --predic 800 --initial "disquieting earth"


python src/predict.py --session reflect_02 --predic 500 --lines 10 --initial "disquieting earth"


# run the flask app

## cli
python -m flask run --host=0.0.0.0 --port=8002

## as a docker instance
```
docker run -p 8002:8002 --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm  -v $(pwd)/workspace:/workspace textrnn:latest
```






