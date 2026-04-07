VIERNES Cognitive Training Corpus Pipeline

Revised Plan — Web-First Corpus Acquisition

Goal

Construir un pipeline ligero, modular y escalable para recolectar 4 GB – 50 GB de texto técnico usando:

APIs públicas
scraping de documentación
dumps parciales
copia directa de texto web

Todo el texto termina en:

data/corpus_raw/


y luego pasa por limpieza, deduplicación y chunking.

El pipeline no depende exclusivamente de descargar dumps gigantes.

1. Estructura Real del Proyecto
data/

├── corpus_raw/               # texto original sin procesar
│   ├── wikipedia/
│   ├── stackexchange/
│   ├── arxiv/
│   ├── docs/
│   └── books/

├── cleaned/           # texto limpio
├── chunks/            # texto tokenizado
├── datasets/          # dataset final JSONL

├── pipeline/          # scripts del pipeline
│
│   ├── collectors/
│   │   ├── wikipedia_api.py
│   │   ├── stackexchange_api.py
│   │   ├── arxiv_api.py
│   │   └── docs_scraper.py
│
│   ├── processors/
│   │   ├── clean_text.py
│   │   ├── deduplicate.py
│   │   ├── quality_filter.py
│   │   └── chunk_text.py
│
│   ├── utils/
│   │   ├── text_utils.py
│   │   └── dataset_stats.py
│
│   └── run_pipeline.py

2. Fuentes Principales (NO dumps gigantes)

El pipeline prioriza APIs y scraping directo.

Wikipedia API
https://en.wikipedia.org/w/api.php


Ejemplo query:

https://en.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0&prop=extracts&explaintext=1&format=json


Permite descargar artículos uno por uno.

Ventaja:

sin descargar 25GB
texto limpio
StackExchange API
https://api.stackexchange.com/docs


Ejemplo:

https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&site=stackoverflow&filter=withbody


Extrae:

preguntas
respuestas
discusiones técnicas reales

Excelente para entrenar IA conversacional.

arXiv API
http://export.arxiv.org/api/query


Ejemplo:

http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=100


Obtiene:

abstracts
papers técnicos
Documentación técnica (scraping)

El scraper usa:

trafilatura


para extraer solo texto.

Fuentes recomendadas:

https://docs.python.org
https://developer.mozilla.org
https://pytorch.org/docs
https://numpy.org/doc
https://huggingface.co/docs
https://kubernetes.io/docs
https://nodejs.org/docs

Libros técnicos libres
https://www.gutenberg.org
https://standardebooks.org
https://openstax.org


Muchos en texto plano.

3. Flujo del Pipeline
collect
↓
raw text
↓
clean
↓
deduplicate
↓
chunk
↓
dataset

4. Recolección (collectors)

Cada collector guarda texto en:

data/raw/<source>/


Ejemplo:

data/raw/wikipedia/wiki_001.txt
data/raw/stackexchange/se_001.txt
data/raw/arxiv/arxiv_001.txt

5. Limpieza

clean_text.py

elimina HTML
normaliza espacios
elimina scripts
elimina tablas inútiles
6. Filtro de calidad

quality_filter.py

descarta:

textos < 500 caracteres
spam
páginas vacías
páginas con exceso de código
7. Deduplicación

deduplicate.py

usa:

MinHash


para eliminar:

duplicados
páginas clonadas
contenido repetido
8. Chunking

chunk_text.py

divide documentos en fragmentos:

500 – 1500 tokens


Formato final:

JSONL


ejemplo:

{"text":"..."}
{"text":"..."}
{"text":"..."}

9. Objetivo de tamaño
Wikipedia API      ~2GB
StackExchange API  ~1GB
arXiv              ~500MB
Docs scraping      ~500MB
Books              ~500MB


Resultado:

4GB+ corpus

10. Ejecución
python data/pipeline/run_pipeline.py collect
python data/pipeline/run_pipeline.py clean
python data/pipeline/run_pipeline.py build
