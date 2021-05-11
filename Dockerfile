FROM python:3.8.0-buster

ADD app.py .

RUN pip install -r requirements.txt

CMD ["python", "./app.py"]
