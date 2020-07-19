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
