FROM python:3.10.0-slim

RUN mkdir /app

WORKDIR /app

COPY . /app/

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]