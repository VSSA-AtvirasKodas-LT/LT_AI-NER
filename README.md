# MLKV VMS

MLKVM modelio kokybės validavimas pagal vieną iš GLUE (angl. General Language Understanding Evaluation) vertinimo metodikoje (https://gluebenchmark.com/) numatytų vertinimo užduočių: įvardytų esybių atpažinimas.

## Modelis

Validuoti modeliai:
- ModernBERT Stage 3 RC 1: [neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1](https://huggingface.co/neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1)

## Reikalavimai

- python3.12
- make
- Huggingface sesija (prisijungiant su huggingface-cli komanda)

### Skaičiavimo resursų reikalavimai

Atlikti skaičiavimams reikia 8GB vaizdo plokštės atminties.

## Duomenų paruošimas

Norint pradėti dirbti su duomenimis, mum reikia prieigos prie:
- Lithuanian jsonl Named Entity Recognition duomenų rinkinio su mokinimo ir testavimo duomenimis (https://github.com/tilde-nlp/MultiLeg-dataset/tree/main, direktorijos data/lt/);

Duomenys turi būti atsiųsti į direktorijas:
- data/lt_test
- data/lt_train

Automatiškai galima parsisiųsti duomenis naudojant komandą:
```bash
make getdata
```

### Duomenų pavertimas iš `jsonl` formato į `conll` formatą

Modelį apmokinant įvardytų esybių atpažinimo uždaviniui naudojamas Conll formatas. Kadangi mūsų duomenys yra `jsonl` formato, juos pasiverčiame į `conll` formatą naudodami `src/utils/jsonl_converter.py` kodą.

Norint atlikti pavertimą, paleidžiame kodą:

```bash
python src/utils/prepare_jsonl.py
```

Norint atsisiųsti `jsonl` formato duomenis ir automatiškai juos pasiversti į `conll` formatą, naudojame komandą:
```bash
make prepare_data
```

## Programos paleidimas

Sukuriame virtualią python aplinką ir įdiegiame naudojamas bibliotekas su komanda:
```bash
make prepare_python
```

Parsisiunčiame duomenis, juos paruošiame, paleidžiame modelio apmokinimo kodą ir atliekame įvertinimą su komanda:
```bash
make all
```

### Modelio apmokinimas

Modelį apmokiname įvardytų esybių atpažinimo uždaviniui ir atliekame įvertinimą su komanda:
```bash
make finetune_modernbertRC1
```

## Skaičiuojami įvertinimo rodikliai

_Token-Level:_
- Accuracy
- Precision
- Recall
- F1-score

_Entity-Level:_
- Exact Match: Full span and type must match
- Overlap Match: Any token overlap with same label counts
- Union Match: Prediction overlaps in any way with true entity

_Each of these includes:_
- Precision
- Recall
- F1-score