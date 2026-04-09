import json

# Start with our base commodities and indices
tickers = {
    # Indices
    "NIFTY 50 (Index)": "^NSEI",
    "BSE SENSEX (Index)": "^BSESN",
    "NIFTY BANK (Index)": "^NSEBANK",
    
    # Global Commodities
    "Gold (COMEX)": "GC=F",
    "Silver (COMEX)": "SI=F",
    "Crude Oil WTI (Petrol/Diesel indicator)": "CL=F",
    "Brent Crude Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Aluminum": "ALI=F",
    
    # US Tech
    "Apple Inc.": "AAPL",
    "Microsoft Corp.": "MSFT",
    "Google (Alphabet)": "GOOGL",
    "Tesla Inc.": "TSLA",
    "Amazon.com Inc.": "AMZN",
    "NVIDIA Corp.": "NVDA"
}

# Now load the NSE stocks we downloaded
try:
    with open('nse_stocks.json', 'r') as f:
        nse_stocks = json.load(f)
    
    # Combine them (our manual ones take precedence of being at the top, since python >3.7 dicts preserve order)
    # Actually let's group them or just put them all in one dict
    for k, v in nse_stocks.items():
        if k not in tickers:
            tickers[k] = v
            
except Exception as e:
    print(f"Error loading nse_stocks.json: {e}")

with open('comprehensive_tickers.json', 'w') as f:
    json.dump(tickers, f, indent=4)
    
print(f"Successfully created comprehensive_tickers.json with {len(tickers)} assets.")
