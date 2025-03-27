import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.jse.co.za"
INDICES_URL = "https://www.jse.co.za/indices"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://www.jse.co.za/",
    "Accept-Language": "en-US,en;q=0.9"
}

session = requests.Session()
session.headers.update(HEADERS)

def get_company_links():
    """Récupère les liens de toutes les entreprises listées sur la page principale."""
    try:
        response = session.get(INDICES_URL, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion : {e}")
        return {}

    if "captcha" in response.text.lower():
        print("Le site semble bloquer les requêtes automatisées (Captcha détecté).")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    company_links = {}

    for row in soup.select("table tbody tr"):  
        try:
            symbol = row.select_one("td:nth-child(1)").text.strip()
            name = row.select_one("td:nth-child(2)").text.strip()
            link = row.select_one("td a")["href"]
            company_links[symbol] = BASE_URL + link if link.startswith("/") else link
        except (AttributeError, TypeError):
            continue

    return company_links

def get_stock_data(company_url):
    """Récupère les données OPEN, HIGH, LOW, CLOSE d'une entreprise."""
    try:
        response = session.get(company_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion pour {company_url} : {e}")
        return []

    if "captcha" in response.text.lower():
        print(f"Le site bloque l'accès à {company_url} (Captcha détecté).")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    stock_data = []

    for row in soup.select("table.daily-stock-data tbody tr"):  
        try:
            date = row.select_one("td:nth-child(1)").text.strip()
            open_price = row.select_one("td:nth-child(2)").text.strip()
            high = row.select_one("td:nth-child(3)").text.strip()
            low = row.select_one("td:nth-child(4)").text.strip()
            close = row.select_one("td:nth-child(5)").text.strip()
            stock_data.append([date, open_price, high, low, close])
        except AttributeError:
            continue

    return stock_data

def main():
    company_links = get_company_links()
    if not company_links:
        print("Aucune entreprise trouvée. Vérifiez la connexion et l'URL.")
        return

    all_data = []

    for symbol, url in company_links.items():
        print(f"Scraping {symbol}...")
        stock_data = get_stock_data(url)
        for row in stock_data:
            all_data.append([symbol] + row)
        time.sleep(2)  # Évite d'envoyer trop de requêtes d'un coup

    if all_data:
        df = pd.DataFrame(all_data, columns=["Symbol", "Date", "Open", "High", "Low", "Close"])
        df.to_csv("jse_stock_data.csv", index=False)
        print("Données enregistrées dans jse_stock_data.csv")
    else:
        print("Aucune donnée collectée.")

if __name__ == "__main__":
    main()
