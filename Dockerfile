FROM python:3.11

#install enviroment 
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#install directory 
WORKDIR /app


COPY requirements.txt .

#install all package 
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY ./development/backend/ .

CMD ["/bin/bash"]