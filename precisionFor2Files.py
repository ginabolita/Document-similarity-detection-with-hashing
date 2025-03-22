
import pandas as pd
import os
import sys

#Devuelve valores de la 3era column (que juesto es Sim% :D)
def get_third_column_values(file_path):
    try:
        df = pd.read_csv(file_path)  
        third_column = df.iloc[:, 2].tolist()  
        return third_column
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 script.py file1.csv file2.csv")
        sys.exit(1)
file_path = sys.argv[1]  # File path1 (Que siempre sea el bruteForce :D)
file_path2 = sys.argv[2]  # File path2 el que comparamos :D

values1 = get_third_column_values(file_path)
values2 = get_third_column_values(file_path2)

#pa que no pete LUEGO MIRAR QUE HACER PARA COMPARACION CON los LSH fores bucket
if len(values1) != len(values2):
        print("Error: The files have different lengths.")
        sys.exit(1)

#calculo de la presicion para cada fila
values = [1 - abs(v1 - v2) for v1, v2 in zip(values1, values2)]
#average precision
prescision = sum(values) / len(values)
print(prescision)

#Ejemplo input: python3 precisionFor2Files.py  bruteForce.csv LSH.csv


