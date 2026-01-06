import json, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# ========= CONFIG =========
# ========= CONFIG =========
SYMBOLS = [
    "ABEV3.SA",
    "ALOS3.SA",
    "ASAI3.SA",
    "AURE3.SA",
    "AXIA3.SA",
    "AXIA6.SA",
    "AXIA7.SA",
    "AZZA3.SA",
    "B3SA3.SA",
    "BBAS3.SA",
    "BBDC3.SA",
    "BBDC4.SA",
    "BBSE3.SA",
    "BEEF3.SA",
    "BPAC11.SA",
    "BRAP4.SA",
    "BRAV3.SA",
    "BRKM5.SA",
    "CEAB3.SA",
    "CMIG4.SA",
    "CMIN3.SA",
    "COGN3.SA",
    "CPFE3.SA",
    "CPLE3.SA",
    "CSAN3.SA",
    "CSMG3.SA",
    "CSNA3.SA",
    "CURY3.SA",
    "CXSE3.SA",
    "CYRE3.SA",
    "CYRE4.SA",
    "DIRR3.SA",
    "EGIE3.SA",
    "EMBJ3.SA",
    "ENEV3.SA",
    "ENGI11.SA",
    "EQTL3.SA",
    "FLRY3.SA",
    "GGBR4.SA",
    "GOAU4.SA",
    "HAPV3.SA",
    "HYPE3.SA",
    "IGTI11.SA",
    "IRBR3.SA",
    "ISAE4.SA",
    "ITSA4.SA",
    "ITUB4.SA",
    "KLBN11.SA",
    "LREN3.SA",
    "MBRF3.SA",
    "MGLU3.SA",
    "MOTV3.SA",
    "MRVE3.SA",
    "MULT3.SA",
    "NATU3.SA",
    "PCAR3.SA",
    "PETR3.SA",
    "PETR4.SA",
    "POMO4.SA",
    "PRIO3.SA",
    "PSSA3.SA",
    "RADL3.SA",
    "RAIL3.SA",
    "RAIZ4.SA",
    "RDOR3.SA",
    "RECV3.SA",
    "RENT3.SA",
    "RENT4.SA",
    "SANB11.SA",
    "SBSP3.SA",
    "SLCE3.SA",
    "SMFT3.SA",
    "SUZB3.SA",
    "TAEE11.SA",
    "TIMS3.SA",
    "TOTS3.SA",
    "UGPA3.SA",
    "USIM5.SA",
    "VALE3.SA",
    "VAMO3.SA",
    "VBBR3.SA",
    "VIVA3.SA",
    "VIVT3.SA",
    "WEGE3.SA",
    "YDUQ3.SA",
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
