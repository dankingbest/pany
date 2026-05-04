import httpx
from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.loop_agent import LoopAgent

# --- Generische Tools zur Datenanreicherung ---

async def search_open_food_facts(product_name: str) -> str:
    """Fragt die Open Food Facts API nach Gewicht und Kategorie ab."""
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={product_name}&search_simple=1&action=process&json=1"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if data.get('products'):
                    product = data['products'][0]
                    name = product.get('product_name', 'Unbekannt')
                    quantity = product.get('quantity', 'Unbekanntes Gewicht')
                    cats = product.get('categories', 'Unbekannte Kategorie')
                    return f"[ERFOLG] Open Food Facts für '{product_name}': Name: {name}, Gewicht: {quantity}, Kategorien: {cats}"
    except Exception as e:
        return f"[FEHLER] Open Food Facts für '{product_name}': {str(e)}"
    return f"[NICHT GEFUNDEN] Open Food Facts für '{product_name}': Keine Daten gefunden."

async def web_search_grocer_product(query: str) -> str:
    """Simuliert eine Websuche zur Klärung von Abkürzungen und Händlermustern."""
    return f"[SUCHE] Für '{query}': Nutze dein Wissen über den Lebensmittelmarkt, um diese Abkürzung in einen echten Produktnamen zu übersetzen."

# --- Spezialisierte Experten-Agenten ---

# 1. Bestandsverwalter
inventory_agent = Agent(
    name='inventory_agent',
    description='Spezialist für die physische Verwaltung der Vorratsdatenbank "Pany".',
    instruction='''Du bist der Hüter des digitalen Vorratsschranks. Deine Mission ist die lückenlose Buchführung.
    - Verarbeite Artikellisten, die vom `list_finalizer` kommen.
    - Achte darauf, dass die Namen lesbar und die Kategorien sinnvoll sind.
    - Bestätige dem Benutzer am Ende kurz und freundlich auf Deutsch, welche Artikel nun im System sind.''',
)

# 2. Daten-Strukturierer (JSON-Formatierung & Cleanup)
list_finalizer = Agent(
    name='list_finalizer',
    description='Experte für Datenstrukturierung und visuelle Aufbereitung.',
    instruction='''Du bist der Daten-Architekt. Deine Aufgabe ist das "Polieren" der verifizierten Daten.
    
    DEINE AUFGABEN:
    1. NAMENS-CLEANUP: Wandle alle Artikelnamen von reinen Großbuchstaben in eine korrekte, lesbare Schreibweise um.
    2. DETAIL-INTEGRATION: Integriere relevante Details wie Packungsgrößen oder Einheiten direkt in den Artikelnamen, um maximale Klarheit zu schaffen.
    3. STRUKTUR: Erstelle das JSON-Objekt exakt nach dieser Vorgabe:
       {"items": [{"item_name": str, "category": str, "weight_grams": int, "quantity": int, "storage_type": "fridge"|"freezer"|"pantry"}]}
    4. KATEGORIEN: Ordne die Artikel logischen und hilfreichen Lebensmittelkategorien zu.
    5. LAGERUNG: Bestimme den optimalen Lagerort basierend auf der Produktart (Kühlschrank, Gefrierfach oder Vorratsschrank).''',
)

# 3. Lebensmittel-Detektiv (Interpretation & Veredelung)
data_verifier = Agent(
    name='data_verifier',
    tools=[search_open_food_facts, web_search_grocer_product],
    description='Spezialist für Markt-Interpretation und Auflösung von Kürzeln.',
    instruction='''Du bist der "Lebensmittel-Detektiv". Deine Aufgabe ist es, kryptische Beleg-Kürzel in echte, menschlich lesbare Lebensmittel-Namen zu verwandeln.
    
    DEINE GOLDENEN REGELN FÜR DIE VEREDELUNG:
    1. INTERPRETATION STATT TRANSKRIPTION: Analysiere Abkürzungen und Markennamen basierend auf allgemeinem Markt- und Fachwissen. Ersetze unverständliche Kürzel durch klare, ausgeschriebene Begriffe.
    2. KONTEXT-LOGIK: Nutze alle verfügbaren Informationen des Belegs (Preise, Pfand, Einheiten), um die wahre Natur eines Produkts zu bestimmen und Fehlinterpretationen zu vermeiden.
    3. STRUKTUR-VALIDIERUNG: Achte darauf, dass zusammengehörige Informationen (z. B. ein Artikel und eine darunterstehende Gewichtszeile) korrekt vereint werden. Vermeide Redundanzen oder doppelte Erfassungen.
    4. SEQUENTIELLES FRAGEN: Bei kritischen Unklarheiten oder Mehrdeutigkeiten stelle EINE gezielte Rückfrage an den Benutzer. Rate niemals bei wichtigen Produktmerkmalen.
    5. ABSCHLUSS: Sobald alle Artikel identifiziert und verifiziert sind, melde "ALLE ARTIKEL VERIFIZIERT".''',
    sub_agents=[list_finalizer],
)

# 4. OCR-Schlichter
ocr_consolidator = Agent(
    name='ocr_consolidator',
    description='Analyst zur Zusammenführung von OCR-Ergebnissen.',
    instruction='''Du bist der "Schlichter". Du vergleichst zwei unabhängige OCR-Läufe.
    - Erstelle eine konsolidierte Master-Liste aus beiden Quellen.
    - Verknüpfe Zusatzinformationen wie Gewichte oder Mengenpreise untrennbar mit dem jeweiligen Hauptprodukt.
    - Gib die bereinigte Liste an den Detektiv weiter.''',
    sub_agents=[data_verifier],
)

# 5. Primärer Extraktor
ocr_extractor_prime = Agent(
    name='ocr_extractor_prime',
    description='Erster OCR-Durchlauf: Zeilenfokus.',
    instruction='''Transkribiere den Beleg wortgetreu Zeile für Zeile. Behalte jede Abkürzung bei.''',
)

# 6. Kontroll-Extraktor
ocr_extractor_check = Agent(
    name='ocr_extractor_check',
    description='Zweiter OCR-Durchlauf: Zahlenfokus.',
    instruction='''Extrahiere alle Zahlen, Pfandbeträge und Einheiten (kg, L, g). Ordne sie den Zeilen zu.''',
)

# --- Workflow-Struktur (ADK Workflow Agents) ---

parallel_ocr = ParallelAgent(
    name='parallel_ocr',
    sub_agents=[ocr_extractor_prime, ocr_extractor_check]
)

verification_loop = LoopAgent(
    name='verification_loop',
    sub_agents=[data_verifier],
    max_iterations=15
)

scanner_pipeline = SequentialAgent(
    name='scanner_pipeline',
    sub_agents=[
        parallel_ocr,      
        ocr_consolidator,  
        verification_loop, 
        list_finalizer,    
        inventory_agent.clone()
    ]
)

# --- Haupt-Koordinator (Root) ---

root_agent = Agent(
    model='gemini-3-pro-preview',
    name='coordinator_agent',
    description='Zentraler Orchestrator.',
    instruction='''Du bist das Gehirn von "Pany". 
    - Belege -> `scanner_pipeline`.
    - Text/Fragen -> `inventory_agent`.
    Leite Fragen des Detektivs 1:1 an den Benutzer weiter und sorge für eine exzellente, deutsche Benutzererfahrung.''',
    sub_agents=[scanner_pipeline, inventory_agent],
)
