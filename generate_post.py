import os
import datetime
import requests
import time
from urllib.parse import quote
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from markdownify import markdownify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import torch
from duckduckgo_search.exceptions import DuckDuckGoSearchException

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Configurazione
ENERGY_TOPICS = [
    "mobilità sostenibile", "batterie al litio", "fotovoltaico", 
    "energia rinnovabile", "bollette energia", "punti ricarica elettrica",
    "accumulo energetico", "smart grid", "comunità energetiche",
    "GSE", "fonti rinnovabili", "transizione energetica"
]
MODEL = "gpt2"
IMAGE_API = "https://source.unsplash.com/random/800x400/?{query}"
MAX_SOURCES = 3
TIMEOUT_SECONDS = 10


def select_energy_topic():
    """Seleziona l'argomento energetico più attuale"""
    with DDGS() as ddgs:
        topic_ranking = []
        for topic in ENERGY_TOPICS:
            results = list(ddgs.text(f"{topic} novità 2024", max_results=2))
            topic_ranking.append((topic, len(results)))
        
        return max(topic_ranking, key=lambda x: x[1])[0]


def fetch_energy_content(topic):
    """Recupera contenuti tecnici specifici"""
    sources = []
    query = f"{topic} sito:gse.it OR sito:enea.it OR sito:ministero-ambiente.it OR filetype:pdf"
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SOURCES))
            sources.extend([r["href"] for r in results])
    except DuckDuckGoSearchException as e:
        print(f"Errore nella ricerca: {e}")
    
    content = ""
    for url in sources:
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS, verify=False)
            if url.endswith('.pdf'):
                content += f"PDF tecnico: {url}\n\n"
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            content += f"## Fonte: {url}\n\n{markdownify(text[:1000])}\n\n"
        except requests.Timeout:
            print(f"Timeout superato per {url}, saltato.")
        except Exception as e:
            print(f"Errore processando {url}: {e}")
    
    return content


def generate_technical_article(topic, sources_text):
    """Genera contenuto tecnico accurato"""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    prompt = f"""
    Scrivi un articolo tecnico su "{topic}" basandoti sulle seguenti fonti:
    {sources_text[:2000]}
    
    Struttura l'articolo come segue:
    1. **Introduzione**: Breve panoramica sul tema.
    2. **Dati chiave**: Informazioni tecniche e numeri rilevanti.
    3. **Normative e innovazioni recenti**: Analisi degli sviluppi normativi e tecnologici.
    4. **Prospettive future**: Implicazioni e sviluppi attesi.
    
    Articolo:
    """
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    max_length = model.config.max_length
    
    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length-10]
    
    try:
        output = model.generate(input_ids, max_length=max_length, pad_token_id=model.config.eos_token_id)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.split("Articolo:")[-1].strip()
    except Exception as e:
        print(f"Errore nella generazione: {e}")
        return "Errore nella generazione dell'articolo."


def save_energy_article(topic, content):
    """Salva con formattazione specifica per energia"""
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
        print("Nessun contenuto tecnico trovato")
        exit(1)
        
    article = generate_technical_article(topic, content)
    saved_file = save_energy_article(topic, article)
    
    print(f"Articolo generato: {saved_file}")
    print(f"Anteprima:\n{article[:200]}...")
