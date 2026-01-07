import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

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
OUT_CSV = None  # ex.: "data/marketdata.csv"


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _append_nulls(results, batch):
    """Se um batch falhar, adiciona linhas com null para não quebrar o pipeline."""
    for sym in batch:
        results.append(
            {
                "symbol": sym,
                "price": None,
                "vol_annual": None,
            }
        )


def _extract_close_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai a matriz de fechamentos (Close) no formato:
      index = datas
      colunas = tickers
    Suporta o formato que o yfinance devolve para 1 ticker ou vários tickers.
    """
    # Caso o df venha vazio
    if df is None or df.empty:
        return pd.DataFrame()

    # Caso comum com group_by="column": df["Close"] funciona se existir
    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()
        return close

    # Caso alternativo: MultiIndex nas colunas (ex.: ('Close', 'PETR4.SA'))
    if isinstance(df.columns, pd.MultiIndex):
        # tenta achar o nível "Close"
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0, drop_level=True)
            if isinstance(close, pd.Series):
                close = close.to_frame()
            return close

    # Se não conseguiu extrair
    return pd.DataFrame()


def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    results = []

    for batch in chunked(SYMBOLS, 100):
        try:
            df = yf.download(
                tickers=batch,
                period=LOOKBACK,
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as e:
            print(f"[WARN] Batch falhou (exception): {e}")
            _append_nulls(results, batch)
            continue

        close = _extract_close_df(df)

        # Se não veio nada de Close / veio vazio, não quebra: registra null e segue
        if close.empty or len(close.index) == 0:
            print("[WARN] Batch retornou vazio/sem Close. Registrando nulls.")
            _append_nulls(results, batch)
            continue

        # Garante que todas as colunas estejam presentes (se alguns tickers falharam)
        # cria colunas faltantes com NaN
        for sym in batch:
            if sym not in close.columns:
                close[sym] = np.nan

        # Ordena colunas para consistência (opcional)
        close = close[batch]

        # Último preço conhecido por ticker
        close_ffill = close.ffill()
        # Se por algum motivo ainda não tiver linha depois do ffill, evita iloc[-1]
        if close_ffill.empty or len(close_ffill.index) == 0:
            print("[WARN] Close após ffill ficou vazio. Registrando nulls.")
            _append_nulls(results, batch)
            continue

        last_price = close_ffill.iloc[-1]

        # Vol anualizada (log-retornos)
        logret = np.log(close / close.shift(1))
        vol = logret.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)

        for sym in batch:
            p = last_price.get(sym, np.nan)
            v = vol.get(sym, np.nan)

            results.append(
                {
                    "symbol": sym,
                    "price": None if pd.isna(p) else float(round(float(p), 6)),
                    "vol_annual": None if pd.isna(v) else float(round(float(v), 8)),
                }
            )

    payload = {
        "generated_at_utc": now,
        "source": "yfinance",
        "interval": INTERVAL,
        "lookback": LOOKBACK,
        "trading_days": TRADING_DAYS,
        "count": len(results),
        "data": results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if OUT_CSV:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8")

    ok_prices = sum(1 for r in results if r["price"] is not None)
    ok_vols = sum(1 for r in results if r["vol_annual"] is not None)
    print(f"OK: atualizado {OUT_JSON} com {len(results)} tickers.")
    print(f"   Preços OK: {ok_prices} | Vols OK: {ok_vols}")


if __name__ == "__main__":
    main()
