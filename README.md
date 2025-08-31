# bayes-speech-classifier
Implementacija Bayesovog klasifikatora za segmente signala govora korištenjem VEPRAD baze.  
Projekt uključuje ekstrakciju MFCC značajki, klasifikaciju fonema pomoću Gaussian Naive Bayes modela te evaluaciju uspješnosti kroz *classification report* i *confusion matrix*.

## Preuzimanje podataka
Zbog veličine, podaci nisu uključeni direktno u repozitorij.  
Dataset (VEPRAD, sample `sm_04`) dostupan je na sljedećem linku:  
[Preuzmite podatke s Google Drive-a](https://drive.google.com/drive/folders/1NwUY405qsrME7epf1xc2lh0kjDiFmon2?usp=sharing)

## Struktura repozitorija
```
bayes-speech-classifier/
├─ src/
│  └─ VEPRAD__PROJEKT.py   # glavni Python skript
├─ requirements.txt        # Python zavisnosti
├─ README.md
```


Nakon preuzimanja, raspakirajte sadržaj u direktorij `data/` unutar repozitorija tako da struktura izgleda ovako:
```
data/
├─ sm_04_wav/    # .wav datoteke
└─ sm_04_lab/    # .lab datoteke
```

## Pokretanje

Pokrenite glavni skript iz root direktorija repozitorija:

```bash
python src/bayes_classifier.py


