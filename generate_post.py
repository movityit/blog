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
    retries = 3  # Number of retries for rate limit errors
    query = f"{topic} sito:gse.it OR sito:enea.it OR sito:ministero-ambiente.it OR filetype:pdf"

    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=MAX_SOURCES))
                sources.extend([r["href"] for r in results])
                if len(sources) >= MAX_SOURCES:
                    break
        except DuckDuckGoSearchException as e:
            print(f"Rate limit error: {e}. Retrying ({attempt+1}/{retries})...")
            time.sleep(5)  # Wait before retrying
        except Exception as e:
            print(f"Errore processando query: {e}")
            break

    if len(sources) < 2:
        for attempt in range(retries):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(f"{topic} tecnico approfondito", max_results=MAX_SOURCES))
                    sources.extend([r["href"] for r in results][:MAX_SOURCES-len(sources)])
                    if len(sources) >= MAX_SOURCES:
                        break
            except DuckDuckGoSearchException as e:
                print(f"Rate limit error: {e}. Retrying ({attempt+1}/{retries})...")
                time.sleep(5)  # Wait before retrying
            except Exception as e:
                print(f"Errore processando query: {e}")
                break

    content = ""
    for url in sources:
        try:
            response = requests.get(url, timeout=8, verify=False)  # Disable SSL verification
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
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    prompt = f"""Genera un articolo tecnico su {topic} con:
1. Dati precisi e verificati
2. Analisi degli sviluppi recenti
3. Tecnologie coinvolte
4. Riferimenti normativi (se applicabile)

Fonti ufficiali:
{sources_text[:2000]}

Articolo completo:"""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    max_length = model.config.max_length

    # Truncate the input to fit within the model's maximum length
    if input_ids.size(1) > max_length:
        truncated_prompt = prompt[:max_length - 10]  # Leave some space for the model to generate text
        input_ids = tokenizer.encode(truncated_prompt, return_tensors='pt')
    
    try:
        output = model.generate(input_ids, max_length=max_length, pad_token_id=model.config.eos_token_id)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.split("Articolo completo:")[-1].strip()
    except (IndexError, ValueError) as e:
        print(f"Error: {e}")
        print(f"Prompt: {prompt}")
        return "Errore nella generazione dell'articolo. Riprovare con un argomento diverso o verificare il modello GPT."
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
