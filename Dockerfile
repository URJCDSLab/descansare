FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN  pip install .
RUN python src/models/train_model.py

WORKDIR /usr/src/app/dashboard
CMD [ "python", "app.py" ]
