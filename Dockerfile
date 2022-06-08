FROM pytorch/torchserve:0.6.0-cpu
ENV USER model-server
ENV HOME /home/${USER}
WORKDIR ${HOME}
COPY *.py ${HOME}/
COPY pyproject.toml ${HOME}
COPY poetry.lock ${HOME}
COPY poetry.toml ${HOME}
COPY config.properties ${HOME}
COPY ast2vec.pt ${HOME}
RUN pip install poetry
RUN poetry install
RUN torch-model-archiver --model-name ast2vec --version 1.0\
        --model-file ast2vec.py --serialized-file ast2vec.pt\
        --export-path model-store\
        --handler handler:entry_point_function_name
