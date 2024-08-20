FROM python:3.12

ARG REMOTE_USER=fred
ARG REMOTE_GID=1000
ARG REMOTE_UID=1000

RUN apt update && apt install pandoc -y 

RUN addgroup --gid ${REMOTE_GID} ${REMOTE_USER} && adduser --disabled-password --uid ${REMOTE_UID} --gid ${REMOTE_GID} ${REMOTE_USER}

USER ${REMOTE_USER}
