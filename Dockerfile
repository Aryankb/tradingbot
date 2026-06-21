# Pull the official, lightweight Python 3.12 image
FROM python:3.12-slim

# Install system utilities required for compiling ML packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# Copy your trading bot code files into the container
COPY . .

# Launch your main loop script
CMD ["python", "bot.py"]
