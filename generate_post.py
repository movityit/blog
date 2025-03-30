import os
import datetime
import requests
from urllib.parse import quote
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from markdownify import markdownify
from transformers import pipeline

# Configurazione
ENERGY_TOPICS = [
    "mobilità sostenibile", "batterie al litio", "fotovoltaico", 
    "energia rinnovabile", "bollette energia", "punti ricarica elettrica",
    "accumulo energetico", "smart grid", "comunità energetiche",
    "GSE", "fonti rinnovabili", "transizione energetica"
]
MODEL = "gpt2"
IMAGE_API = "https://source.unsplash.com/random/800x400/?{query}"
MAX_SOURCES = 3  # Numero fonti da consultare

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
    
    # Ricerca specializzata
    query = f"{topic} sito:gse.it OR sito:enea.it OR sito:ministero-ambiente.it OR filetype:pdf"
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=MAX_SOURCES))
        sources.extend([r["href"] for r in results])
    
    # Fallback per contenuti tecnici
    if len(sources) < 2:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{topic} tecnico approfondito", max_results=MAX_SOURCES))
            sources.extend([r["href"] for r in results][:MAX_SOURCES-len(sources)])
    
    content = ""
    for url in sources:
        try:
            response = requests.get(url, timeout=8)
            if url.endswith('.pdf'):
                content += f"PDF tecnico: {url}\n\n"
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            content += f"## Fonte: {url}\n\n{markdownify(text[:800])}\n\n"
        except Exception as e:
            print(f"Errore processando {url}: {e}")
    
    return content

def generate_technical_article(topic, sources_text):
    """Genera contenuto tecnico accurato"""
    generator = pipeline("text-generation", model=MODEL)
    
    prompt = f"""Genera un articolo tecnico su {topic} con:
1. Dati precisi e verificati
2. Analisi degli sviluppi recenti
3. Tecnologie coinvolte
4. Riferimenti normativi (se applicabile)

Fonti ufficiali:
{sources_text[:2000]}

Articolo completo:"""
    
    result = generator(
        prompt,
        max_length=1200,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.6  # Più conservativo per contenuti tecnici
    )
    
    return result[0]["generated_text"].split("Articolo completo:")[-1].strip()

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
    
    os.makedirs("articoli_energia", exist_ok=True)
    with open(f"articoli_energia/{filename}", "w", encoding='utf-8') as f:
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
