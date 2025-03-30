import os
import datetime
import random
import requests
import nltk
from bs4 import BeautifulSoup
from markdownify import markdownify
from duckduckgo_search import DDGS
from transformers import pipeline

# Scarica modelli NLP (solo al primo avvio)
nltk.download("punkt")

# Carica il modello di linguaggio open-source
text_generator = pipeline("text-generation", model="distilgpt2")

# Lista di argomenti generici
topics = [
    "mobilità elettrica", "energia rinnovabile", "intelligenza artificiale", "criptovalute",
    "spazio e astronomia", "salute e benessere", "cambiamento climatico", "biotecnologie",
    "robotica", "storia medievale", "arte e design", "psicologia", "filosofia", "imprenditoria",
    "scienza dei materiali", "alimentazione sostenibile", "energia nucleare", "Internet delle cose",
    "batterie al litio", "geopolitica", "quantum computing", "educazione digitale"
]

# Scegli 10 argomenti casuali
chosen_topics = random.sample(topics, 10)

# Crea una cartella per gli articoli se non esiste
os.makedirs("_posts", exist_ok=True)

for topic in chosen_topics:
    print(f"Generando articolo su: {topic}")

    # Cerca su DuckDuckGo
    query = f"{topic} ultime notizie"
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    # Estrai contenuti da massimo 3 fonti
    extracted_content = ""
    for result in results:
        url = result["href"]
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            content = markdownify(soup.get_text())
            extracted_content += content[:1000] + "\n\n"  # Prende massimo 1000 caratteri per fonte
        except:
            print(f"Errore nel recupero dati da: {url}")

    if not extracted_content:
        print(f"Nessuna informazione trovata per {topic}, salto...")
        continue

    # Riscrive il testo in modo originale con l'IA
    prompt = f"Scrivi un articolo informativo su {topic}. Usa un linguaggio chiaro e accessibile:\n{extracted_content[:1500]}"
    generated_text = text_generator(prompt, max_length=800, num_return_sequences=1)[0]["generated_text"]

    # Crea il file Markdown per Jekyll
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"_posts/{date}-{topic.replace(' ', '-')}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"---\ntitle: {topic.capitalize()}\ndate: {date}\ncategories: {topic.replace(' ', '-')}\n---\n\n")
        f.write(generated_text)

    print(f"Articolo creato: {filename}")
