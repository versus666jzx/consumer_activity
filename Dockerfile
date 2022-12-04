FROM python:3.10
RUN apt update && apt-get -y upgrade && \
    /usr/local/bin/python -m pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir && \
    rm requirements.txt && \
    mkdir /opt/consumer_activity
COPY src/ /opt/consumer_activity/src
COPY data/ /opt/consumer_activity/data
WORKDIR /opt/consumer_activity
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port", "80"]