import os
from nltk.corpus import reuters

# Directorio donde se guardarán los archivos
output_dir = 'reuters_articles'
os.makedirs(output_dir, exist_ok=True)

# Seleccionar una categoría con suficientes artículos
category = 'earn'  # 'earn' es una de las categorías más grandes

# Obtener los IDs de los archivos en la categoría seleccionada
file_ids = reuters.fileids(category)

# filtrar los archivos por longitud
file_ids = [file_id for file_id in file_ids if len(reuters.raw(file_id)) > 2000]

# Limitar a los primeros 100 artículos
file_ids = file_ids[:100]

# Guardar cada artículo en un archivo .txt
for file_id in file_ids:
    # Obtener el contenido del artículo
    article_text = reuters.raw(file_id)
    
    # Crear un nombre de archivo basado en el ID del archivo
    file_name = f"{file_id.replace('/', '_')}.txt"
    file_path = os.path.join(output_dir, file_name)
    
    # Guardar el contenido en un archivo .txt
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(article_text)

print(f"Se han guardado {len(file_ids)} artículos en el directorio '{output_dir}'.")
