import subprocess
import csv
import argparse

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

def main():
    # Configura el parser de argumentos
    parser = argparse.ArgumentParser(description='Document Similarity Methods Evaluation')
    parser.add_argument('--k_values', nargs='+', type=int, required=True, help='List of k values to test')
    parser.add_argument('--num_docs', type=int, default=20, help='Number of documents to generate')
    
    # Parsea los argumentos
    args = parser.parse_args()

    # Ejecuta el primer programa (exp1_genRandPerm)
    print("Ejecutando exp1_genRandPerm...")
    salida_programa1 = ejecutar_programa("exp1_genRandPerm", args.num_docs)
    if salida_programa1 is None:
        return  # Termina el script si hay un error

    # Ejecuta el segundo programa (exp2_genRandShingles) para cada valor de k
    print("Ejecutando exp2_genRandShingles...")
    for k in args.k_values:
        salida_programa2 = ejecutar_programa("exp2_genRandShingles", k, args.num_docs)
        if salida_programa2 is None:
            continue  # Continúa con el siguiente valor de k si hay un error

        # Procesa la salida de exp2_genRandShingles (aquí puedes personalizar)
        print(f"Salida para k={k}: {salida_programa2}")

    # Aquí puedes agregar código para procesar las salidas y generar un archivo CSV
    # Por ejemplo:
    # guardar_csv(datos, "resultados.csv")

if __name__ == "__main__":
    main()