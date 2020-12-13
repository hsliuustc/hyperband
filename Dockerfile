From mindspore/mindspore-cpu:1.0.1

# set the working directory
WORKDIR /code 

# copy the dependencies file to the working directory

COPY requirements.txt . 

# install dependencies
RUN pip install -r requirements.txt

# copy the local files to the working directory
# COPY mindspore-example/ .
