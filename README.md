# MLKVM validavimo sprendimas

MLKVM kokybės validavimas pagal vieną iš GLUE (angl. General Language Understanding Evaluation) vertinimo metodikoje (https://gluebenchmark.com/) numatytų vertinimo užduočių: įvardytų esybių atpažinimas.

## Modelis

Šiame sprendime buvo validuojamas ModernBERT architektūros bendrasis vektorinis modelis: [VSSA-SDSA/LT-MLKM-modernBERT](https://huggingface.co/VSSA-SDSA/LT-MLKM-modernBERT)

Remiantis šiuo sprendimu NER užduočiai pritaikytą modelį galite rasti: [VSSA-SDSA/LT-NER-modernBERT](https://huggingface.co/VSSA-SDSA/LT-NER-modernBERT)

## Reikalavimai

- python3.12
- make
- Huggingface sesija (prisijungiant su huggingface-cli komanda)

### Skaičiavimo resursų reikalavimai

Atlikti skaičiavimams reikia 8GB vaizdo plokštės atminties.

## Programos paleidimas

Norint paleisti programą, reikia suskurti virtualią aplinką ir į ją įrašyti reikiamas naudojamų bibliotekų versijas. Tai galima padaryti su komanda:
```bash
make prepare_python
```

Modelį pavyks paleisti tik pasiruoštoje virtualioje aplinkoje su nurodytomis bibliotekų versijomis.Norint paleisti visą programą, pirmiausia parsisiunčiame duomenis, juos paruošiame, paleidžiame modelio apmokinimo kodą ir atliekame įvertinimą. Visą tai galime atlikti su viena komanda:
```bash
make all
```

Programą galime paleisti ir pažingsniui. Norint tai padaryti, sekite žemiau pateiktus žingsnius.

### Duomenų paruošimas

Norint pradėti dirbti su duomenimis, mum reikia prieigos prie:
- Lithuanian jsonl Named Entity Recognition duomenų rinkinio su mokinimo ir testavimo duomenimis (https://github.com/tilde-nlp/MultiLeg-dataset/tree/main, direktorijos data/lt/);

Duomenys turi būti atsiųsti į direktorijas:
- data/lt_test
- data/lt_train

Duomenis galima parsisiųsti automatiškai naudojant komandą:
```bash
make getdata
```

Modelį apmokinant įvardytų esybių atpažinimo uždaviniui naudojamas Conll formatas. Kadangi mūsų duomenys yra `jsonl` formato, juos pasiverčiame į `conll` formatą naudodami `src/utils/jsonl_converter.py` kodą.

Norint atlikti pavertimą, paleidžiame kodą:

```bash
python src/utils/prepare_jsonl.py
```

Norint visus duomenų paruošimo žingsius atlikti pasinaudojus viena komanda, naudojame komandą:
```bash
make prepare_data
```

### Modelio pritaikymas

Modelį apmokiname įvardytų esybių atpažinimo uždaviniui ir atliekame įvertinimą su komanda:
```bash
make finetune_modernbertRC1
```

## Modelio naudojimas

Modelio testavimo skriptai yra pateikti `inference_rc1.ipynb` faile. Ten rasite modelio naudojimo DEMO ir tuo pačiu galėsite ištestuoti savo apmokytus modelius.


## Skaičiuojami įvertinimo rodikliai

Kiekvienas modelis apmokymo metu yra įvertinamas šiomis metrikomis:

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