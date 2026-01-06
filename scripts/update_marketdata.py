import json, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# ========= CONFIG =========
SYMBOLS = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    # ... coloque o resto aqui
]
LOOKBACK = "400d"
INTERVAL = "1d"
TRADING_DAYS = 252

OUT_JSON = "data/marketdata.json"
OUT_CSV  = None  # troque para "data/marketdata.csv" se quiser CSV

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    results = []
    for batch in chunked(SYMBOLS, 100):
        df = yf.download(
            tickers=batch,
            period=LOOKBACK,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )

        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()

        last_price = close.ffill().iloc[-1]
        logret = np.log(close / close.shift(1))
        vol = logret.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)

        for sym in batch:
            p = last_price.get(sym, np.nan)
            v = vol.get(sym, np.nan)

            results.append({
                "symbol": sym,
                "price": None if pd.isna(p) else float(round(p, 6)),
                "vol_annual": None if pd.isna(v) else float(round(v, 8)),
            })

    payload = {
        "generated_at_utc": now,
        "source": "yfinance",
        "interval": INTERVAL,
        "lookback": LOOKBACK,
        "trading_days": TRADING_DAYS,
        "data": results
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if OUT_CSV:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"OK: atualizado {OUT_JSON} com {len(results)} tickers.")

if __name__ == "__main__":
    main()
