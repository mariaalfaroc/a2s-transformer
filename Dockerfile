FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

RUN apt update --fix-missing
RUN apt install build-essential -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y
RUN apt update --fix-missing
RUN apt install fluidsynth -y
RUN apt install git -y
RUN apt clean

RUN pip install --upgrade pip
RUN pip install pybind11

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U datasets