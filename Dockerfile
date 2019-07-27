FROM nvcr.io/nvidia/pytorch:19.06-py3
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt