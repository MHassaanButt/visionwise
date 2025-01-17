FROM python:3.9-slim

RUN mkdir /app

COPY * /app/

WORKDIR /app

RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

RUN pip install -r requirements.txt

CMD ["python", "app.py"]

