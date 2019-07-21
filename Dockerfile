FROM python:3.6

RUN pip install keras tensorflow numpy Pillow

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip install -r requirements.txt

COPY . /

ENTRYPOINT ["python3" ]

CMD [ "./main.py" ]


