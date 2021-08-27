FROM nvcr.io/nvidia/pytorch:20.06-py3

# WORKDIR /usr/src/app

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

RUN apt-get update \
    && apt-get install -y \
        nmap \
        vim

# COPY . /app
# WORKDIR /app
# RUN pip3 install --upgrade pip
# RUN pip3 install -r requirements.txt 
# EXPOSE 5001


# ENTRYPOINT ["flask"]
# CMD [ "run", "--host=0.0.0.0" ]


WORKDIR /usr/src/app

RUN apt-get update -y
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8002"]

