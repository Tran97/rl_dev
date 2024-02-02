###############################################################################################
# Base Image for #
###############################################################################################
ARG ROS_DISTRO=humble

FROM osrf/ros:${ROS_DISTRO}-desktop-full as base
SHELL ["/bin/bash", "-c"]

# Install basic apt packages
RUN apt-get update && apt-get install -y \
  git \
  python3-pip \
  #python3-tk \
  wget \
  vim \
  swig


WORKDIR /dev_ws
COPY 02465students/ .
#Install requirements
RUN pip install -r requirements_conda.txt --no-cache-dir
RUN pip install -r requirements_pip.txt --no-cache-dir


# Remove display warnings
RUN mkdir /tmp/runtime-root
ENV XDG_RUNTIME_DIR "/tmp/runtime-root"
ENV NO_AT_BRIDGE 1


# Dev container arguments
ARG USERNAME=devuser
ARG UID=1000
ARG GID=${UID}


# Create new user and home directory
RUN groupadd --gid $GID $USERNAME \
 && useradd --uid ${GID} --gid ${UID} --create-home ${USERNAME} \
 && mkdir -p /etc/sudoers.d \
 && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME} \
 && mkdir -p /home/${USERNAME} \
 && chown -R ${UID}:${GID} /home/${USERNAME}

# Set the ownership of the workspace to the new user
RUN chown -R ${UID}:${GID} /dev_ws/

# Set the user and source entrypoint in the user's .bashrc file
USER ${USERNAME}

RUN echo "/home/steven/.local/bin" >> /home/${USERNAME}/.bashrc