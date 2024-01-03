FROM public.ecr.aws/lambda/python:3.9


COPY requirements.txt .
<<<<<<< HEAD
COPY nltkdata ./nltkdata
COPY app/* ${LAMBDA_TASK_ROOT}
# COPY app/* .
RUN pip install --no-cache-dir --upgrade -r requirements.txt  --target "${LAMBDA_TASK_ROOT}" 
# RUN python -m nltk.downloader punkt
# RUN python -m nltk.downloader stopwords

CMD [ "main.handler" ]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "79"]
=======
# ARG PIP_DISABLE_PIP_VERSION_CHECK=1
# ARG PIP_NO_CACHE_DIR=1
RUN pip3 install -r requirements.txt --target ${LAMBDA_TASK_ROOT}
COPY ./app/* ${LAMBDA_TASK_ROOT}
CMD [ "main.handler" ]
>>>>>>> parent of 2c79f66 ("fix some issues")
