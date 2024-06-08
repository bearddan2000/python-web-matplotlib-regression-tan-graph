FROM python

WORKDIR /code

COPY bin .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# CMD "python app.py"
