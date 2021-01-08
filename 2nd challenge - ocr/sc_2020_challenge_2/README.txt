Sadrzaj ovog foldera upload-ovati u GoogleDrive repozitorijum koji ste dobili u pozivnici za nedeljni izazov 1.


Da biste pokrenuli rešenje na svojoj mašini i proverili kolika je postignuta tačnost, potrebno je uraditi sledeće:

1. Implementirati metodu u process.py traženom logikom. Ovaj fajl ne pokrećete direktno.
2. Pokrenuti main.py (iz pycharm-a na Run, ili iz terminala komandon “python main.py” uz prethodno aktiviranje odgovarajućeg virtuelnog okruženja). Pokretanje main.py fajla će izgenerisati result.csv fajl, tako što će pozvati prethodno implementiranu metodu za sve primerke iz skupa podataka.
3. Pokrenuti evaluate.py fajl (iz pycharm-a na Run, ili iz terminala komandon “python evaluate.py” uz prethodno aktiviranje odgovarajućeg virtuelnog okruženja). Ovaj fajl će učitati result.csv koji je prethodno generisan i izračunati tačnost. Izlaz ovog fajla je samo broj koji pokazuje procenat tačnosti trenutnog rešenja.




"serialized_model" folder:
U ovaj folder sacuvati serijalozovani model u slucaju da ste koristili serijalizaciju.

Platforma ne vrsi download "dataset" foldera, posto njega studenti nece modifikovati (statican je), ali ce vrsiti download "serialized_model" foldera. Zbog toga je u njega potrebno sacuvati fajlove vec istreniranih modela.
