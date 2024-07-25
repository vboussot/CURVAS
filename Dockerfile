FROM --platform=linux/amd64 pytorch/pytorch
    
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user data.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
