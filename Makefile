VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip


all: prepare_data finetune_modernbertRC1

prepare_python:
	python3.12 -m venv $(VENV)
	$(PIP) install -r requirements.txt

prepare_data: makedirs getdata
	$(PYTHON) src/utils/prepare_jsonl.py

clean:
	rm -rf data
	rm -rf output_modernbert_rc1

makedirs:
	mkdir -p data

getdata:
	git clone git@github.com:tilde-nlp/MultiLeg-dataset.git
	cp -r MultiLeg-dataset/data/lt/test data/lt_test
	cp -r MultiLeg-dataset/data/lt/train data/lt_train
	rm -rf MultiLeg-dataset

finetune_modernbertRC1:
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/finetune.py --config config/modernbert-RC1.yml
