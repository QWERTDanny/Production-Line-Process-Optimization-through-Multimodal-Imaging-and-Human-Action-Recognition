FROM cyrilbq/torch1.13-cu11.6-cudnn8-git:base

COPY AVer .
RUN pip install pytorchvideo

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt