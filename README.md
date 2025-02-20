# Intro

In einem Vespa-Container wird der extrahierte Text der englischen PDFs des MiR200 Roboters in vier verschiedenen Formaten abgelegt und kann durchsucht werden.
> [Spring direkt zum Set-Up](#anleitung-zur-einrichtung-und-verwendung-von-vespa-lokal-in-einem-docker-container)

Die verschiedenen Formate sind:

1. **ocr**  
   *Nur Text-Extraktion (Plain Text).*

2. **markdown**  
   *OCR mit Layout-Erkennung; das Ergebnis wird als Markdown ausgegeben.*

3. **chunked_ocr**  
   *OCR mit `RecursiveCharacterTextSplitter` (LangChain).  
   Chunk Size: 500 Characters, Overlap: 50 Characters.*

4. **chunked_markdown**  
   *Markdown mit `MarkdownTextSplitter` (LangChain).  
   Chunk Size: 500 Characters, Overlap: 50 Characters.*

_(Siehe [Notebook `transformation.ipynb`](./transformation.ipynb) für weitere Details zum Chunking und [Notebook `extraction.ipynb`](./extraction.ipynb) für den Code für das PDF-Processing.)_

---

## Embedding-Modell

- Ein neues, kleineres Embedding-Modell (float statt bfloat32) wurde verwendet.  
- Das Modell läuft schneller und liefert ausreichend gute Ergebnisse.  
- Model und Tokenizer liegen als ONNX-Datei (framework-agnostisch, single file) im `model`-Ordner.  
- Diese werden zusammen mit der Vespa-App deployed.

> **ONNX-Transformation**  
>  
> ```bash
> ./export_hf_model_from_hf.py \
>   --hf_model intfloat/multilingual-e5-small \
>   --output_dir model \
>   --patch_tokenizer
> ```
>  
> - **`--hf_model`**: Name oder Pfad des Hugging-Face-Modells (z.B. `intfloat/multilingual-e5-small`).  
> - **`--output_dir`**: Zielverzeichnis für das konvertierte ONNX-Modell.   
> - **`--patch_tokenizer`**: Patcht den Tokenizer, falls spezielle Anpassungen erforderlich sind.
> - **`--quantize`**: Aktiviert die Quantisierung (verkleinert das Modell und beschleunigt ggf. die Inferenz). Wurde nicht verwendet, aber bietet Möglichkeit Modelle noch kleiner und schneller zu machen

---

# Vespa Informationen

Vespa ist in erster Linie eine **Open Text Search Engine**, die Vektor-Suche ermöglicht. Sie bietet jedoch eine Menge zusätzlicher Funktionalitäten und ist sehr skalierbar. Im Folgenden ein Überblick:

## Vorteile von Vespa

- **Suche**:  
  Vespa ist eine echte Text-Suchmaschine und bietet umfangreiche Customization-Optionen, die über einfache Vektor-Datenbanken hinausgehen.  

- **BM25 und Wort-basierte Suche**:  
  Anders als bei vielen Large-Language-Model-Anwendungen wird kein fixes Vocabulary benötigt. Stattdessen wird über Stemming (z.B. *haben* → *hab*) und Normalisierung (z.B. *à* → *a*) gearbeitet. Dies ermöglicht eine exaktere Übereinstimmung mit den Begriffen der jeweiligen Domäne.  

- **Deployment des Embedding-Modells**:  
  Das Embedding-Modell kann direkt mit der Vespa App deployt werden. Das vermeidet unnötigen Datentransfer zwischen Datenbank und Client.  

- **Automatische Spracherkennung**:  
  Ab mehr als drei Wörtern wird eine Sprache erkannt und dann je nach Sprache getokenized (Stemming, Normalisierung).  

- **Integrierte Retrieval- und Ranking-Phasen**:  
  Beide laufen direkt in Vespa und können so effizienter genutzt werden. Eine externe Implementierung (z.B. beim Client, wie etwa bei Qdrant) ist nicht notwendig.  

## Aufbau einer Vespa-Applikation

- **Services und Schema**:  
  Eine Vespa-App besteht aus mindestens einer `services.xml` und einer Schema-Spezifikation (`schema`-Ordner).  
  - In der `services.xml` wird u.a. definiert:  
    - Mindest-Redundanz  
    - Embedding-Modell  
  - Im `schema`-Ordner werden die Felder (Dokumentstruktur) definiert, z.B.:
    - Welche Felder sollen Indexing mit Stemming/Normalisierung erhalten (`index`)?
    - Welche Felder sind nur zur Filterung oder Sortierung (`attribute`)?
    - Welche Felder sollen in Summaries zurückgegeben werden (`summary`)?

- **Indexing-Einstellungen**:
  - `set_language` – basierend auf diesem Feld kann Vespa die Sprache bestimmen.  
  - `index` – für unstrukturierte Strings (textbasierte Felder mit BM25 und Vektor-Suche).  
  - `attribute` – für strukturierte Daten, die gefiltert, gruppiert und sortiert werden.  
  - `summary` – bestimmt, welche Felder in den Ergebnis-Snippets zurückgegeben werden.

- **Verschiedene Matching-Modi**:
  - `text`: Wortbasiert mit Stemming/Normalisierung (für Volltextsuche, BM25).  
  - `word`: Genaues Matching ohne Stemming (für strukturierte Abfragen, z.B. Filter).  
  - `exact`: Genaues Matching wie `word` nur ohne spezielle Sonderzeichen

- **Document-Summaries**:
  - Definieren unterschiedliche Sichten (Profile) auf ein Dokument, z.B. ob nur Titel und Kurzzusammenfassung zurückgegeben werden sollen.

- **Embedding-Feld**:
  - F32-Tensor mit 384 Dimensionen.  
  - Distanzmetrik: *angular* (entspricht Cosine Similarity).

- **Fieldset**:
  - Gruppen von Feldern, auf die bei Retrieval und Ranking gemeinsam zugegriffen werden soll.

- **Ranking-Profiles**:
  - Mehrere Phasen (First Phase, Second Phase, Global Phase) zur Dokument-Ranking und Sortierung.  
  - Sparse Scores (z.B. BM25) können addiert werden oder mit Dense Scores gemischt werden (verschiedene Fusionsverfahren).

---

## Retrieval und Ranking in Vespa

1. **Retrieval**:  
   - Identifiziert aus **allen** Dokumenten jene, die für die Anfrage relevant sein könnten.  
   - Schnelle Filter, BM25-Suche und `nearestNeighbor`-Vektor-Suche.

2. **Ranking**:  
   - Ordnet die Menge der abgerufenen Dokumente erneut nach komplexeren Kriterien.  
   - Mehrere Stufen:  
     1. First Phase: Basierend auf Node-spezifischen Faktoren.  
     2. Global Phase: Auswertung aller Dokumente, die von den Nodes geliefert werden.

In einem Single-Node-Setup unterscheidet sich die Global Phase nicht wesentlich – die Daten liegen ohnehin nur auf einer Node.

---

## Hybrid-Fusion: Sparse und Dense

Vespa bietet mehrere Optionen zur Kombination (Fusion) von BM25 (Sparse) und Embedding-Ähnlichkeit (Dense):

- **Reciprocal Rank Fusion (hybrid-rrf)**  
  Führt eine Fusion basierend auf den Rängen der Dokumente durch.  
- **Density-Based Scoring Fusion**  
  Normalisiert die Scores der Dokumente und addiert sie.

Da hierfür alle Dokument-Scores nötig sind (von den Dokumenten, die noch übrig sind), werden die Fusionen ausschließlich in der globalen Ranking-Phase durchgeführt.

---
# Anleitung zur Einrichtung und Verwendung von Vespa lokal in einem Docker-Container

## 1. Download Vespa Image und Vespa CLI

```bash
docker pull vespaengine/vespa
brew install vespa-cli
```

## 2. Starten und Konfigurieren von Vespa

**Hinweis:** Der Test-Container benötigt mindestens 6 GB RAM. Im Beispiel verwenden wir 8 Cores, was nicht zwingend notwendig ist, aber praktisch für Entwicklungszwecke.

### Container starten

```bash
docker run --detach --name waku-search --hostname vespa-container \
  --publish 8080:8080 \
  --publish 19071:19071 \
  vespaengine/vespa
```

- Port **8080**: Data-Plane (für das Schreiben und Abfragen von Dokumenten).  
- Port **19071**: Control-Plane (für das Deployen der Anwendung).  

Beachte, dass der Port 8080 **erst nach dem Deployment** der Anwendung aktiv ist.

### Vespa CLI konfigurieren

Setze das Target auf den lokalen Container:

```bash
vespa config set target local
```

### Service-Status überprüfen

Das Starten des Containers kann etwas dauern. Vergewissere dich, dass der Konfigurationsservice läuft:

```bash
vespa status deploy --wait 300
```

Alternativ im Browser:  
[http://localhost:19071/state/v1/health](http://localhost:19071/state/v1/health)

## 3. Anwendung deployen

Navigiere in den `waku-search` Ordner und starte die Anwendung:

```bash
vespa deploy app
```

**Tipp:** Überprüfe die Docker-Logs oder rufe [http://localhost:8080/state/v1/health](http://localhost:8080/state/v1/health) auf, um sicherzugehen, dass alles fehlerfrei gestartet ist (Suche nach `"completed successfully"` in den Logs).

## 4. Daten aus Ordner importieren (Feed)

```bash
vespa feed -t http://localhost:8080 vespa-feed/*.json
```

## 5. Die Vespa-Search Container-Abfragen

### 5.1 Attribute Matching

```bash
curl -X POST http://localhost:8080/search/ \
  -H 'Content-Type: application/json' \
  -d '{
    "yql": "select * from doc where text_format contains \"ocr\"",
    "hits": 5,
    "language": "en"
  }'
```

### 5.2 Semantic Search

```bash
curl -X POST http://localhost:8080/search/ \
  -H 'Content-Type: application/json' \
  -d '{
    "yql": "select * from doc where text_format contains \"ocr\" and ({targetHits:100}nearestNeighbor(embedding,e))",
    "query": "What steps are required to reactivate a battery in DeepSleep?",
    "input.query(e)": "embed(@query)",
    "ranking.profile": "semantic",
    "hits": 5
  }'
```

### 5.3 Text-based Search (z.B. BM25)

Wenn Sprache nicht angegeben wird sie automatisch bestimmt. 

```bash
curl -X POST http://localhost:8080/search/ \
  -H 'Content-Type: application/json' \
  -d '{
    "yql": "select * from doc where text_format contains \"ocr\" and (userInput(@query))",
    "query": "whats the plan out of deep sleep",
    "ranking.profile": "bm25",
    "hits": 5,
    "language": "en"
  }'
```

### 5.4 Hybrid Search

```bash
curl -X POST http://localhost:8080/search/ \
  -H 'Content-Type: application/json' \
  -d '{
    "yql": "select * from doc where text_format contains \"ocr\" and (({targetHits:100}nearestNeighbor(embedding,e)) or (userInput(@query)))",
    "query": "whats the plan out of deep sleep",
    "input.query(e)": "embed(@query)",
    "hits": 5,
    "ranking.profile": "hybrid-rrf",
    "language": "en"
  }'
```

### 5.5 Exact Search Mode
specify field set which can be searched as one, in unserem Fall nicht relevant, weil seite soll nur kommen wenn auf ihr wirklich das gesuchte steht
```bash
curl -X POST http://localhost:8080/search/ \
  -H 'Content-Type: application/json' \
  -d '{
    "yql": "select * from doc where text_format contains \"ocr\" and (text contains @query)",
    "query": "undergo maintenance",
    "hits": 5,
    "language": "en"
  }'
```

#### 5. 6 Logische Operatoren (OR, AND)


> **Überlegung:** Der Nutzer kann entweder eine normale Query eingeben (Hybrid Pipeline) oder den Exact Search Mode verwenden, welcher durch Anführungszeichen oder spezifische Syntax ausgelöst wird. Dort kann man logische Operatoren wie AND/OR nutzen. Für eine kombination von exakter und normaler Suche muss Ranking für das Filtering im Feld `text` ausgeschaltet werden, da es sonst darauf ranked .


### Exact Search
```bash
# OR
curl -X POST http://localhost:8080/search/ \
  -H 'Content-Type: application/json' \
  -d '{
    "yql": "select * from doc where text_format contains \"ocr\" and (text contains @query or text contains @query1)",
    "query": "deep sleep",
    "query1": "charge",
    "hits": 5,
    "language": "en"
  }'

# AND
curl -X POST http://localhost:8080/search/ \
  -H 'Content-Type: application/json' \
  -d '{
    "yql": "select * from doc where text_format contains \"ocr\" and (text contains @query and text contains @query1)",
    "query": "deep sleep",
    "query1": "charge",
    "hits": 5,
    "language": "en"
  }'
```

### Kombination von normaler und exact Search

```bash
curl -X POST http://localhost:8080/search/ \
  -H 'Content-Type: application/json' \
  -d '{
    "yql": "select * from doc where text_format contains \"ocr\" and ((({targetHits:100}nearestNeighbor(embedding,e)) or (userInput(@query))) and text contains ({ranked:false}@query1))",
    "query": "whats the plan out of deep sleep",
    "query1": "charging",
    "input.query(e)": "embed(@query)",
    "hits": 5,
    "ranking.profile": "hybrid-rrf",
    "language": "en"
  }'
```

## 6. Änderung in der `app`-Konfiguration und Redeploy

Wenn im `app`-Verzeichnis Änderungen (z.B. an Schemas, Ranking-Profiles etc.) vorgenommen wurden, muss erneut deployt werden:

```bash
vespa deploy app
```

Solange das Deployment läuft, ist die bisherige Konfiguration weiterhin aktiv (keine Downtime).

## 7. Evaluation

### Beispielaufruf eines Evaluierungs-Skripts

```bash
python evaluation_ranking.py --mode hybrid --ranking hybrid-rrf --input qrels.json --format "ocr"
```

**Parameter**:  
- **`--mode`**: Gibt an, welche Retrieval-Variante verwendet werden soll (z.B. `hybrid`, `semantic`, `dense`).  
- **`--ranking`**: Welches Ranking-Profil genutzt werden soll (z.B. `hybrid-rrf`, `bm25`, etc.).  
- **`--input`**: Pfad zur Evaluationsdatei, in der die relevanten Dokument-IDs und Queries gespeichert sind (z.B. `qrels.json`).  
- **`--format`**: Welches Textformat evaluiert werden soll (z.B. `ocr`, `markdown`,`chunked_markdowwn`,`chunked_ocr` ).  

Die verwendeten Metriken (z.B. NDCG, P, R) sind im Code hardcoded, lassen sich jedoch leicht anpassen.  

## 8. Weitere Möglichkeiten

- **Cross Encoder Model** integrieren für ein finales Reranking der letzten 10 Dokumente
- **Highlighting**: Markiere gefundene Passagen im Text (Vespa bietet rudimentäre Text-Highlighting-Funktionen).  
- Chunks einer Seite als **ein Dokument** speichern: (Array of strings und embeddings als Array of tensors) kann Overhead reduzieren und einfacher für Client.  
- **OR** und **AND** für Standard Retrieval Mode einsetzen.  

## 9. Mehr Informationen

- [https://docs.vespa.ai/en/tutorials/text-search.html](https://docs.vespa.ai/en/tutorials/text-search.html)  
- [https://docs.vespa.ai/en/tutorials/hybrid-search.html](https://docs.vespa.ai/en/tutorials/hybrid-search.html)  
- [https://github.com/vespa-engine/sample-apps/tree/master/simple-semantic-search](https://github.com/vespa-engine/sample-apps/tree/master/simple-semantic-search)  
- [https://github.com/vespa-engine/sample-apps](https://github.com/vespa-engine/sample-apps)  

