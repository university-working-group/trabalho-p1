FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models /app/logs /app/face_database

RUN wget -q https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2 -O /app/models/shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 /app/models/shape_predictor_68_face_landmarks.dat.bz2

COPY facial_recognition_system.py /app/
COPY facial_recognition_headless.py /app/

COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["gui"]