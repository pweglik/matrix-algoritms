### Lab 1 algorytmy: 
 * Strassen
 * Binet
 * Winograd

Wszystkie są podobne różnią się tylko operacjami. Winograd w dodatku paskudnie wygląda bo próbowałem go zoptymalizować. 

Wszystkie oznaczenia i informacje brałem z:
https://en.wikipedia.org/wiki/Strassen_algorithm

### Pomocnicze funkcje:
* split - dzieli podaną macierz na mniejsze o rozmiarach podanych w parametrach
* resize_matrix_to_2n - zmienia rozmiar macierzy poprzez dodawanie zer w taki sposób aby ilość kolumn jak i wierszy była potęgą dwójki
* find_next_power_of_2 - znajduje najbliższą większą od podanej liczby potęgę dwójki
* Counter - klasa przez którą przepuszczam operacje aby je zliczać pewnie jest jakiś sprytniejszy sposób jak jakiś dekorator, ale już mi się nie chciało xD
