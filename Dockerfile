FROM python:3.7

RUN apt-get update \
    && apt-get install -y \
        nmap \
        vim

COPY . /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt 
EXPOSE 5001


ENTRYPOINT ["flask"]
CMD [ "run", "--host=0.0.0.0" ]