FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN apt-get update && \
apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# EXPOSE 80

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /usr/src/app/src
COPY data/ /usr/src/app

# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["uvicorn", "src.main:app"]