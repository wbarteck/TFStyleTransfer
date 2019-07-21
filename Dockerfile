FROM tensorflow/tensorflow:latest-py3

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip install -r requirements.txt

COPY . /

ENTRYPOINT ["python3" ]

CMD [ "./main.py" ]


