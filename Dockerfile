FROM python:3.8
COPY requirements.txt .
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN pip3 install -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]
