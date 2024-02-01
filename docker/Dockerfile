FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /dev_ws

# # Dev container arguments
# ARG USERNAME=devuser
# ARG UID=1000
# ARG GID=${UID}

# # Install extra tools for development
# RUN apt-get update && apt-get install -y --no-install-recommends \
#  gdb gdbserver nano

# # Create new user and home directory
# RUN groupadd --gid $GID $USERNAME \
#  && useradd --uid ${GID} --gid ${UID} --create-home ${USERNAME} \
#  && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
#  && chmod 0440 /etc/sudoers.d/${USERNAME} \
#  && mkdir -p /home/${USERNAME} \
#  && chown -R ${UID}:${GID} /home/${USERNAME}

# # Set the ownership of the overlay workspace to the new user
# RUN chown -R ${UID}:${GID} /overlay_ws/ /underlay_ws/


# # Set the user and source entrypoint in the user's .bashrc file
# USER ${USERNAME}