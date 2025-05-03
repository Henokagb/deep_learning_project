FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install fastapi
RUN pip install transformers 
RUN pip install uvicorn
RUN pip install torch


EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]