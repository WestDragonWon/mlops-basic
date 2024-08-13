#경량화된 파이썬 이미지를 사용하여 도커 이미지를 생성한다.
FROM python:3.11-alpine3.19 

LABEL maintainer="westdragonwon"

# 컨테이너에 찍히는 로그를 볼 수 있도록 허용한다.
ENV PYTHONUNBUFFERED 1

# tmp -> 최대한 컨테이너를 경량화하기 위해 빌드가 완료된 후에 사용한 파일을 삭제한다.
COPY ./requirements.txt /tmp/requirements.txt
COPY ./requirements.dev.txt /tmp/requirements.dev.txt
COPY ./app /app

WORKDIR /app
EXPOSE 8000


RUN python -m venv /py && \ 
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /tmp/requirements.txt && \
    if [ $DEV = "true" ]; \
        then /py/bin/pip install -r /tmp/requirements.dev.txt ; \
    fi && \
    rm -rf /tmp && \
    adduser \
        --disabled-password \
        --no-create-home \
        django-user

ENV PATH="/py/bin:$PATH"

USER django-user

# Django(Scikit-learn => REST API) - Docker - Github Actions(CI/CD)

ARG DEV=false