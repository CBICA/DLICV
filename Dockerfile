# Use an official Python runtime as a parent image
FROM python:3.8-slim

LABEL IMAGE="DLICV"

RUN mkdir /DLICV/  && pip install DLICV==1.0.0

# Download the model zip file
ADD https://github.com/CBICA/DLICV/releases/download/v0.0.0/model.zip /DLICV/

# Unzip the model and remove the zip file
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/* && \
    unzip /DLICV/model.zip -d /DLICV/ && \
    rm /DLICV/model.zip

# Run DLICV.py when the container launches
# Note: The entrypoint uses the model path inside the container.
# Users can mount their own model at /DLICV/model/ if they want to use a different one.
ENTRYPOINT ["DLICV", "--model", "/DLICV/model/"]
CMD ["--help"]