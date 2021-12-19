## Start from this Docker image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

## Install Python packages in Docker image
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

## Copy your files into Docker image
COPY e1d3/utils /home/eelab/Desktop/brats-e1d3/e1d3/utils
COPY e1d3/train.py /home/eelab/Desktop/brats-e1d3/e1d3/
COPY e1d3/test.py /home/eelab/Desktop/brats-e1d3/e1d3/
COPY e1d3/docker_supervise.py /home/eelab/Desktop/brats-e1d3/e1d3/

## Make Docker container executable
ENTRYPOINT ["python", "/home/eelab/Desktop/brats-e1d3/e1d3/docker_supervise.py"]
