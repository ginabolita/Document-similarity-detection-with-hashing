import subprocess
import csv
import argparse
import os

def ejecutar_programa(nombre_programa, *args):
    """
    Ejecuta un programa en C++ con los argumentos proporcionados y captura su salida.
    """
    # Construye el comando con los argumentos
    comando = [f"./{nombre_programa}"] + [str(arg) for arg in args]
    
    # Ejecuta el programa
    resultado = subprocess.run(comando, capture_output=True, text=True)
    
    # Verifica si hubo un error
    if resultado.returncode != 0:
        print(f"Error al ejecutar {nombre_programa}: {resultado.stderr}")
        return None
    
    # Devuelve la salida estándar
    return resultado.stdout

def crear_carpeta_results():
    """
    Crea la carpeta 'results' si no existe.
    """
    if not os.path.exists("results"):
        os.makedirs("results")

def crear_csv_vacio(nombre_archivo):
    """
    Crea un archivo CSV vacío en la carpeta 'results'.
    """
    with open(f"results/{nombre_archivo}", "w") as archivo:
        pass  # Solo crea el archivo vacío

def guardar_csv(datos, nombre_archivo):
    """
    Guarda los datos en un archivo CSV en la carpeta 'results'.
    """
    with open(f"results/{nombre_archivo}", "a", newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(datos)

def main():
    # Configura el parser de argumentos
    parser = argparse.ArgumentParser(description='Document Similarity Methods Evaluation')
    parser.add_argument('--k_values', nargs='+', type=int, required=True, help='List of k values to test')
    parser.add_argument('--b_values', nargs='+', type=int, required=True, help='List of b values to test')
    parser.add_argument('--num_docs', type=int, default=20, help='Number of documents to generate')
    parser.add_argument('--num_docs2', type=int, default=30, help='Number of documents to generate')
    
    # Parsea los argumentos
    args = parser.parse_args()

    # Crea la carpeta 'results' y los archivos CSV vacíos
    crear_carpeta_results()
    crear_csv_vacio("exp1_genRandPerm_results.csv")
    crear_csv_vacio("exp2_genRandShingles_results.csv")
    crear_csv_vacio("jaccardBruteForce_results.csv")
    crear_csv_vacio("jaccardMinHash_results.csv")
    crear_csv_vacio("jaccardLSHbase_results.csv")
    crear_csv_vacio("jaccardLSHbucketing_results.csv")
    crear_csv_vacio("jaccardLSHforest_results.csv")

    # Ejecuta el primer programa (exp1_genRandPerm) desde el directorio actual
    print("Ejecutando exp1_genRandPerm...")
    salida_programa1 = ejecutar_programa("exp1_genRandPerm", args.num_docs)
    if salida_programa1 is None:
        return  # Termina el script si hay un error

    # Ejecuta el segundo programa (exp2_genRandShingles) desde el directorio actual para cada valor de k
    print("Ejecutando exp2_genRandShingles...")
    for k in args.k_values:
        salida_programa2 = ejecutar_programa("exp2_genRandShingles", k, args.num_docs2)
        if salida_programa2 is None:
            continue  # Continúa con el siguiente valor de k si hay un error

        # Procesa la salida de exp2_genRandShingles (aquí puedes personalizar)
        print(f"Salida para k={k}: {salida_programa2}")

    # Ejecuta jaccardLSHbucketing y jaccardLSHforest para cada combinación de k y b
    print("Ejecutando jaccardLSHbucketing y jaccardLSHforest...")
    for k in args.k_values:
        for b in args.b_values:
            # Ejecuta jaccardLSHbucketing
            salida_bucketing = ejecutar_programa("jaccardLSHbucketing", "exp1_directory", k, b)
            if salida_bucketing is not None:
                # Guarda la salida en el archivo CSV correspondiente
                guardar_csv([k, b, salida_bucketing], "jaccardLSHbucketing_results.csv")
                print(f"Salida para jaccardLSHbucketing (k={k}, b={b}): {salida_bucketing}")

            # Ejecuta jaccardLSHforest
            salida_forest = ejecutar_programa("jaccardLSHforest", "exp1_directory", k, b)
            if salida_forest is not None:
                # Guarda la salida en el archivo CSV correspondiente
                guardar_csv([k, b, salida_forest], "jaccardLSHforest_results.csv")
                print(f"Salida para jaccardLSHforest (k={k}, b={b}): {salida_forest}")

if __name__ == "__main__":
    main()