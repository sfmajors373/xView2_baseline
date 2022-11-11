FROM python:3.8
COPY requirements.txt .
RUN apt-get update -y
RUN apt-get install libgl1-mesa-glx -y

# Install GDAL dependencies
RUN apt-get install -y libgdal-dev

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip install GDAL==3.2.2.1
RUN pip3 install -r requirements.txt
COPY . .
RUN mkdir -p working_dir
WORKDIR /working_dir
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--app-dir", "/"]
