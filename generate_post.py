import os
import datetime
import random
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from markdownify import markdownify

# Argomenti casuali per gli articoli
topics = ["mobilità elettrica", "energia rinnovabile", "incentivi auto elettriche", "batterie al litio", "colonnine di ricarica"]

# Scegli un argomento casuale
topic = random.choice(topics)

# Cerca su Google
query = f"{topic} ultime notizie"
results = list(search(query, num=5, stop=5, lang="it"))

# Prendi il primo risultato valido
url = results[0]

# Scarica il contenuto
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Estrai il testo
content = markdownify(soup.get_text())

# Crea il file Markdown
date = datetime.datetime.now().strftime("%Y-%m-%d")
filename = f"_posts/{date}-{topic.replace(' ', '-')}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"---\ntitle: {topic.capitalize()}\ndate: {date}\ncategories: {topic.replace(' ', '-')}\n---\n\n")
    f.write(content[:2000])  # Limita la lunghezza dell'articolo

print(f"Creato articolo: {filename}")
