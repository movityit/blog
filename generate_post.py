import os
import datetime
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from duckduckgo_search import DDGS
from transformers import pipeline
from urllib.parse import quote

# Configurazione open-source
nltk.download("punkt")  # Solo al primo avvio
MODEL = "gpt2"  # Modello open-source (alternativa: "EleutherAI/gpt-neo-125M")
IMAGE_API = "https://source.unsplash.com/random/800x400/?{query}"

# Argomenti possibili
topics = [
    "mobilità elettrica", "energia rinnovabile", "intelligenza artificiale",
    "quantum computing", "cambiamento climatico", "open source software"
]

def get_most_relevant_topic():
    """Seleziona l'argomento con più risultati recenti"""
    with DDGS() as ddgs:
        topic_scores = []
        for topic in topics:
            results = list(ddgs.text(f"{topic} ultime notizie 2024", max_results=5))
            topic_scores.append((topic, len(results)))
        return max(topic_scores, key=lambda x: x[1])[0]

def fetch_open_content(topic):
    """Recupera contenuti da fonti aperte (licenze libere)"""
    sources = [
        "https://it.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles={topic}&explaintext=1",
        f"https://api.creativecommons.engineering/v1/images?q={quote(topic)}&license_type=commercial,modification"
    ]
    content = ""
    try:
        # Wikipedia (CC BY-SA)
        wiki_res = requests.get(sources[0].format(topic=topic), timeout=5).json()
        page = next(iter(wiki_res["query"]["pages"].values()))
        content += page.get("extract", "")[:1000] + "\n\n"
        
        # Openverse (API immagini CC)
        img_res = requests.get(sources[1], timeout=5).json()
        cover_url = img_res["results"][0]["url"] if img_res["results"] else IMAGE_API.format(query=topic)
    except Exception as e:
        print(f"Errore fonti aperte: {e}")
        cover_url = IMAGE_API.format(query=topic)
    return content.strip(), cover_url

def generate_article(topic, source_text):
    """Genera testo originale con modello open-source"""
    generator = pipeline("text-generation", model=MODEL)
    prompt = f"Scrivi un articolo divulgativo su {topic} basato su questi fatti:\n{source_text[:1500]}\n\n---\nArticolo:"
    return generator(
        prompt,
        max_length=1024,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )[0]["generated_text"].split("---\nArticolo:")[-1].strip()

def save_markdown(topic, content, cover_url):
    """Salva in formato Jekyll con metadati"""
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"_posts/{date}-{topic.replace(' ', '-')}.md"
    
    frontmatter = f"""---
title: "{topic.capitalize()}"
date: {date}
categories: ["tech", "innovazione"]
cover: {cover_url}
license: "CC BY-SA 4.0"
---\n\n"""
    
    os.makedirs("_posts", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(frontmatter + content)
    return filename

if __name__ == "__main__":
    topic = get_most_relevant_topic()
    print(f"Generazione articolo su: {topic}")
    
    content, cover_url = fetch_open_content(topic)
    if not content:
        print("Nessun contenuto valido trovato")
        exit(1)
        
    article = generate_article(topic, content)
    saved_file = save_markdown(topic, article, cover_url)
    print(f"Articolo salvato: {saved_file}\nCopertina: {cover_url}")
