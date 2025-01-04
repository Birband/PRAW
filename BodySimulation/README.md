## Wymagania
- SFML - grafika
- OpenMP - dla zrównoleglonych implementacji przy użyciu wątków
- g++ - kompilator

## Uruchamianie projektu
Projekt najpierw musi być skompilowany. W tym celu należy użyć kompilatora g++.
```
g++ nazwa_pliku.cpp -o nazwa_wyjściowa <dodatkowe_flagi>

./nazwa_wyjściowa
```

Dla wersji z OpenMP czyli Prallel trzeba dodać flagę `-fopenmp`.

Dla wersji Visualised trzeba dodać flagi `-lsfml-graphics -lsfml-window -lsfml-system`.

