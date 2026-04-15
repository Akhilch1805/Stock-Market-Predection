import urllib.request
import csv
import json

url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        content = response.read().decode('utf-8')
        
    reader = csv.reader(content.strip().split('\n'))
    header = next(reader)
    symbol_idx = header.index("SYMBOL")
    name_idx = header.index("NAME OF COMPANY")
    
    # Generate the dictionary format for the app
    nse_stocks = {}
    for row in reader:
        try:
            symbol = row[symbol_idx]
            name = row[name_idx]
            nse_stocks[f"{name} ({symbol})"] = f"{symbol}.NS"
        except IndexError:
            pass
            
    with open('nse_stocks.json', 'w') as f:
        json.dump(nse_stocks, f)
        
    print(f"Saved {len(nse_stocks)} NSE stocks to nse_stocks.json")
except Exception as e:
    print(f"Error: {e}")
