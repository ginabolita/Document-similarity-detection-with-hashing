# Detección de similitud de documentos con hashing
## Proyecto de Algorítmia FIB UPC Q2-2024-2025

### Miembros del grupo:
- **Marcel Alabart Benoit**
- **Gina Escofet González**
- **Biel Pérez Silvestre**
- **Albert Usero Martínez**

---

## Descripción
Este proyecto tiene como objetivo principal experimentar con diferentes técnicas para detectar la similitud entre dos documentos. Se utilizan tres métodos principales:
1. **Fuerza bruta**: Cálculo directo de la similitud de Jaccard.
2. **MinHashing**: Aproximación eficiente de la similitud de Jaccard usando hashing.
3. **Locality-Sensitive Hashing (LSH)**: Técnica para reducir el espacio de búsqueda y detectar documentos similares de manera eficiente.

Además, se generan **documentos virtuales** para poder experimentar la efectividad de cada algoritmo. Estos documentos se crean a partir de un texto base al cual se le ha aplicará una normalización. Estudiamos 2 bloques de experimentos principales:
1. **Experimento 1**: Permutación de frases del texto base.
2. **Experimento 2**: Selección aleatoria de grupos de k-shingles del texto base.

---

## Funcionalidad
- **Normalización de textos**:
  - Eliminación de stopwords (palabras que no aportan significado).
  - Conversión de mayúsculas a minúsculas.
- **Cálculo de similitud**:
  - **Fuerza bruta**: Cálculo exacto de la similitud de Jaccard.
  - **MinHashing**: Aproximación de la similitud de Jaccard usando funciones hash.
  - **Locality-Sensitive Hashing (LSH)**: Detección eficiente de documentos similares.
- **Generación de documentos virtuales**:
  - **Experimento 1**: Permutación de frases de un texto base.
  - **Experimento 2**: Selección aleatoria de k-shingles de un texto base.

---

## Instalación
1. Clonar el repositorio:
   ```bash
   git clone git@github.com:Androm3d/Proyecto2.git
   ```
2. Navegar al directorio del proyecto:
   ```bash
   cd Proyecto2
   ```
3. Ejecutar setup.sh:
	```bash
    chmod +x setup.sh
    source setup.sh
	```
---

## Uso
 ### Bloques experimentales
 1. Ejemplo de ejecución completa de experimento 1:
  ```bash
    python3 experimento.py --mode real --experiment_type all --num_docs 40 --prepare_datasets --visualize --num_runs 3 
  ```
2. Ejemplo de ejecución completa de experimento 2:
  ```bash
    python3 experimento.py --mode virtual --experiment_type all --num_docs 40 --prepare_datasets --visualize --num_runs 3 
  ```
### Algoritmos implementados
- **Fuerza bruta**:
  ```bash
  ./jaccardBruteForce <directory> <k>
  ```
	Donde `<k>` es el tamaño de los shingles.

- **MinHashing**:
  ```bash
  ./jaccardMinHash <directory> <k> <t>
  ```
	Donde `<k>` es el tamaño de los shingles.
  Donde `<t>` es el número de funciones hash.


- **LSH bucketing**:
  ```bash
  ./jaccardLSHbucketing  <corpus_dir> <k> <t> <b> <thr>
  ```
	Donde `<k>` es el tamaño de los shingles.
  Donde `<t>` es el número de funciones hash.
	Donde `<b>`es el número de bandas.
  Donde `<thr>` es el umbral.

- **LSH forest**:
  ```bash
  ./jaccardLSHforest <corpus_dir> <k> <t> <b> <thr>
  ```
  Donde `<k>` es el tamaño de los shingles.
  Donde `<t>` es el número de funciones hash.
  Donde `<b>`es el número de bandas.
  Donde `<thr>` es el umbral.

---

## Contacto
- **Marcel Alabart Benoit**: [marcel.alabart@estudiantat.upc.edu](mailto:marcel.alabart@estudiantat.upc.edu)
- **Gina Escofet González**: [gina.escofet@estudiantat.upc.edu](mailto:gina.escofet@estudiantat.upc.edu)
- **Biel Pérez Silvestre**: [biel.perez@estudiantat.upc.edu](mailto:biel.perez@estudiantat.upc.edu)
- **Albert Usero Martínez**: [albert.usero@estudiantat.upc.edu](mailto:albert.usero@estudiantat.upc.edu)

