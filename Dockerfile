FROM nvcr.io/nvidia/isaac-sim:6.0.0-dev2

USER root

# Single RUN to avoid cache issues — install, verify, then build
RUN rm -f /etc/apt/sources.list.d/*.list \
    && echo "deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse" > /etc/apt/sources.list \
    && echo "deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y \
        libpng16-16 \
        libtbb2 \
        libjpeg-turbo8 \
        libturbojpeg \
        libusb-1.0-0 \
        git \
        wget \
        make \
        g++ \
        build-essential \
    && which make \
    && which git \
    && git clone --branch isaacsim-6.0.0 https://github.com/stereolabs/zed-isaac-sim.git /isaac-sim/zed-isaac-sim \
    && cd /isaac-sim/zed-isaac-sim \
    && rm -rf exts/sl.sensor.camera/bin/ \
    && ./build.sh \
    && cp -r exts/sl.sensor.camera /isaac-sim/exts/sl.sensor.camera \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /isaac-sim/zed-isaac-sim/_build /isaac-sim/zed-isaac-sim/_compiler