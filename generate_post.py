import os
import datetime
import requests
import time
from urllib.parse import quote
from bs4 import BeautifulSoup
from markdownify import markdownify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Configurazione
ENERGY_TOPICS = [
    "mobilità sostenibile", "batterie al litio", "fotovoltaico",
    "energia rinnovabile", "bollette energia", "punti ricarica elettrica",
    "accumulo energetico", "smart grid", "comunità energetiche",
    "GSE", "fonti rinnovabili", "transizione energetica"
]
IMAGE_API = "https://source.unsplash.com/random/800x400/?{query}"
MAX_SOURCES = 3
TIMEOUT = 8  # Timeout per le richieste web


def select_energy_topic():
    """Seleziona l'argomento con più risultati su Startpage."""
    topic_ranking = []
    for topic in ENERGY_TOPICS:
        results = fetch_links_from_startpage(f"{topic} novità 2024", max_results=2)
        topic_ranking.append((topic, len(results)))
    return max(topic_ranking, key=lambda x: x[1])[0]


def fetch_links_from_startpage(query, max_results=MAX_SOURCES):
    """Effettua una ricerca su Startpage e restituisce i link trovati."""
    url = f"https://www.startpage.com/do/search?q={quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", class_="w-gl__result-title")]
        return links[:max_results]
    except requests.RequestException as e:
        print(f"Errore Startpage: {e}")
        return []


def fetch_energy_content(topic):
    """Scarica contenuti dai link trovati."""
    sources = fetch_links_from_startpage(f"{topic} approfondimento tecnico")
    content = ""
    
    for url in sources:
        try:
            response = requests.get(url, timeout=TIMEOUT)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            content += f"## Fonte: {url}\n\n{markdownify(text[:800])}\n\n"
        except Exception as e:
            print(f"Errore su {url}: {e}")
    
    return content


def generate_technical_article(topic, sources_text):
    """Genera un articolo basato sui dati trovati."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    prompt = f"""
Scrivi un articolo tecnico su "{topic}" includendo:
- Dati verificati tratti dalle fonti trovate
- Sviluppi recenti e impatti sul settore
- Tecnologie e innovazioni correlate
- Normative vigenti (se disponibili)

Fonti utilizzate:
{sources_text[:2000]}

Articolo:
"""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    max_length = model.config.max_length
    
    if input_ids.size(1) > max_length:
        input_ids = tokenizer.encode(prompt[:max_length - 10], return_tensors='pt')
    
    try:
        output = model.generate(input_ids, max_length=max_length, pad_token_id=model.config.eos_token_id)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.split("Articolo:")[-1].strip()
    except Exception as e:
        print(f"Errore generazione: {e}")
        return "Errore nella generazione dell'articolo."


def save_energy_article(topic, content):
    """Salva l'articolo generato."""
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"energia_{date}_{topic.replace(' ', '_')}.md"
    
    frontmatter = f"""---
title: "{topic.upper()}"
date: {date}
categories: ["energia"]
tags: {[t for t in ENERGY_TOPICS if t in topic.lower()]}
cover: {IMAGE_API.format(query=quote(topic.split()[0]))}
---\n\n"""
    
    os.makedirs("_posts", exist_ok=True)
    with open(f"_posts/{filename}", "w", encoding='utf-8') as f:
        f.write(frontmatter + content)
    
    return filename


if __name__ == "__main__":
    print("Avvio generazione articolo energetico...")
    
    topic = select_energy_topic()
    print(f"Argomento selezionato: {topic}")
    
    content = fetch_energy_content(topic)
    if not content:
        print("Nessun contenuto tecnico trovato. Uscita.")
        exit(1)
        
    article = generate_technical_article(topic, content)
    saved_file = save_energy_article(topic, article)
    
    print(f"Articolo generato: {saved_file}")
    print(f"Anteprima:\n{article[:200]}...")
