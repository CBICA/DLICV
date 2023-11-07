# Use an official Python runtime as a parent image
FROM python:3.8-slim

LABEL IMAGE="DLICV"

RUN mkdir /DLICV/model && pip install DLICV==0.0.0

# Download the model zip file
ADD https://github.com/georgeaidinis/DLICV/releases/download/v0.0.0/model.zip /DLICV/model/

# Unzip the model and remove the zip file
RUN unzip /DLICV/model/model.zip -d /DLICV/model/ && rm /DLICV/model/model.zip

# Run DLICV.py when the container launches
# Note: The entrypoint uses the model path inside the container.
# Users can mount their own model at /DLICV/model/ if they want to use a different one.
ENTRYPOINT ["DLICV", "--model", "/DLICV/model/"]
CMD ["--help"]