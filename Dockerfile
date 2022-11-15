FROM ubuntu:18.04
ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8
RUN apt-get clean && apt-get update -y && apt-get -y install --no-install-recommends apt-utils
RUN apt-get -y install git cmake make
RUN apt-get -y install gcc g++
RUN apt-get -y install protobuf-compiler libprotobuf-dev
RUN apt-get install -y wget
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
RUN bash Anaconda3-2019.03-Linux-x86_64.sh -b
RUN rm Anaconda3-2019.03-Linux-x86_64.sh
ENV PATH /root/anaconda3/bin:$PATH
RUN sh -c echo -e "y\n" 
RUN pip install --upgrade pip
COPY . /linger
WORKDIR /linger
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN chmod +x ./install.sh
RUN ./install.sh