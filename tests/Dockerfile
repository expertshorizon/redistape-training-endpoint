FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY --chown=0:0 test.sh test_outputs.py /tests/
RUN chmod 700 /tests && chmod 700 /tests/test.sh /tests/test_outputs.py

RUN mkdir -p /app && chmod 755 /app

WORKDIR /tests
