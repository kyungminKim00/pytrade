# Rapids + Torch images (cuda 11.2, cudnn8.1)
# FROM ubuntu:18.04
# FROM rapidsai/rapidsai-core:22.08-cuda11.2-runtime-ubuntu18.04-py3.9
FROM nvidia/cuda:11.2
ENV DEBIAN_FRONTEND noninteractive
RUN mkdir /dev_env
WORKDIR /dev_env
COPY . .
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install --yes --no-install-recommends libcudnn8=8.1.0.77-1+cuda11.2
RUN apt-get install -y git
RUN apt-get install -y libcairo2-dev
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata
RUN apt-get install -y vim-nox
RUN apt-get install -y tree
RUN apt-get autoremove -y
RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r /dev_env/requirements.txt
RUN pip3 install --no-cache-dir -r /dev_env/ci_requirements.txt
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu112
# 명시적 선언(컨테이너 생성시 재오픈 필요)
EXPOSE 8888 8787 8786
 
