## Start from this Docker image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

## Install Python packages in Docker image
RUN pip install -r requirements.txt

## Copy your files into Docker image
COPY e1d3/utils /usr/local/bin/utils
COPY e1d3/train.py /usr/local/bin/
COPY e1d3/test.py /usr/local/bin/
COPY e1d3/docker_supervise.py /usr/local/bin/

## Make Docker container executable
ENTRYPOINT ["python", "/usr/local/bin/docker_supervise.py"]