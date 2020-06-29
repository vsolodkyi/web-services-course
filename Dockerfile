FROM python:3.6-slim
COPY . /app
WORKDIR /app
RUN pip install flask gunicorn numpy sklearn scipy flask_wtf pandas




