# Use the official Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Update and upgrade the system
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y tzdata

# Install necessary dependencies for the script
RUN apt-get install -y wget unzip curl software-properties-common lsb-release python3-pip

# Install npm
RUN apt-get install -y npm

# Install Bazelisk using npm
RUN npm install -g @bazel/bazelisk

# Run Bazelisk to install Bazel
RUN bazelisk

# Install Drake's dependencies
RUN wget -q -O /tmp/drake-setup.zip https://github.com/RobotLocomotion/drake/archive/refs/heads/master.zip && \
    unzip -q /tmp/drake-setup.zip -d /tmp && \
    yes | bash /tmp/drake-master/setup/ubuntu/install_prereqs.sh && \
    rm -rf /tmp/drake-setup.zip /tmp/drake-master

# Install ROS Humble
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && \
    apt-get install -y ros-humble-desktop ros-dev-tools python3-rosdep2 python3-colcon-common-extensions

# Initialize rosdep
RUN rosdep init && \
    rosdep update

# Install Drake
ARG DRAKE_RELEASE="20230315"
ARG UBUNTU_VERSION="jammy"
RUN wget -q -O /tmp/drake.tar.gz https://github.com/RobotLocomotion/drake/releases/download/v1.14.0/drake-${DRAKE_RELEASE}-${UBUNTU_VERSION}.tar.gz && \
    tar -xzf /tmp/drake.tar.gz -C /opt && \
    rm /tmp/drake.tar.gz

# Clone drake-ros source code into the container
RUN apt-get update && \
    apt-get install -y git
RUN mkdir -p /ros_ws/src && \
    git clone https://github.com/RobotLocomotion/drake-ros.git /ros_ws/src/drake-ros

# Install source code dependencies
RUN rosdep install -y --from-paths /ros_ws/src --ignore-src --rosdistro humble --skip-keys='drake'

# Build source code
RUN . /opt/ros/humble/setup.sh && \
    cd /ros_ws && \
    colcon build

# Run unit tests and report results
RUN . /opt/ros/humble/setup.sh && \
    cd /ros_ws && \
    colcon test && \
    colcon test-result --verbose

# Set the entrypoint to source ROS and Drake setup files and run a bash shell
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /opt/drake/share/drake/setup.bash &&

