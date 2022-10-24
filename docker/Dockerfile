#  docker build -f docker/Dockerfile -t bpoole908/deep-bci:tf --force-rm --no-cache --build-arg VERSION=tf .
ARG VERSION=tf

FROM bpoole908/mlenv-gpu:${VERSION}

USER root

COPY ./docker/environment.yaml ./setup.py ./

RUN chown $UID:$GID environment.yaml

USER $USER

# Allows each RUN to activate conda env
SHELL ["conda", "run", "-n", "mlenv", "/bin/bash", "-c"]

RUN conda env update --file environment.yaml \
    && conda clean --all -f -y \
    && rm environment.yaml
    
RUN pip install --no-cache-dir -e . \
    && rm setup.py 

CMD /bin/bash

