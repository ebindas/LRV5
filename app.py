import ssl
import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import truststore
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, SecretStr
from starlette.responses import JSONResponse

from notion_client import Client as NotionClient
from kite_trade import KiteApp  # pip install kite-trade

# ================== CONFIG ==================
# Hardcode for local testing ONLY. Move to env vars for deployment.
NOTION_TOKEN = "ntn_hU768711875atGzZiGdD54XmAUxUKK7JWkEQ4efYnXKe0h"
NOTION_SLEEP_SECONDS = 0.34  # ~3 req/sec safe for Notion API

TLS_CTX: ssl.SSLContext = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("app")

notion = NotionClient(auth=NOTION_TOKEN, client=httpx.Client(verify=TLS_CTX))
app = FastAPI(title="Notion ↔ Kite FastAPI", version="local-fund-lots")

# ================== MODELS ==================
class Payload(BaseModel):
    DATABASE_ID: str = Field(
        pattern=r"^[0-9a-f]{32}$",
        description="32-char lowercase hex id",
    )
    Date: date
    entoken: SecretStr
    fund: float = Field(gt=0, description="Capital available for allocation")

# ================== ERROR SHAPE / HEALTH ==================
@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}

@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse({"error": "internal_error"}, status_code=500)

# ================== NOTION HELPERS ==================
def _first_plain_text(prop: Optional[dict], field: str) -> Optional[str]:
    if not prop:
        return None
    arr = prop.get(field, [])
    if not arr:
        return None
    item = arr[0]
    return item.get("plain_text") or item.get("text", {}).get("content")

def fetch_notion_dataframe(database_id: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        kwargs = {"database_id": database_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        resp = notion.databases.query(**kwargs)

        for page in resp.get("results", []):
            props = page.get("properties", {})
            title_arr = props.get("SYMBOL", {}).get("title", [])
            symbol = title_arr[0]["plain_text"] if title_arr else None

            rows.append({
                "pageID": page.get("id"),
                "SYMBOL": symbol,
                "UpperRes": props.get("UpperRes", {}).get("number"),
                "LowerRes": props.get("LowerRes", {}).get("number"),
                "UpperSup": props.get("UpperSup", {}).get("number"),
                "LowerSup": props.get("LowerSup", {}).get("number"),
                "LOT_SIZE": props.get("LOT_SIZE", {}).get("number"),
                "MARGIN_RATE": props.get("MARGIN_RATE", {}).get("number"),
                "MARGIN": props.get("MARGIN", {}).get("number"),
                "instrument_token": _first_plain_text(props.get("instrument_token"), "rich_text"),
            })

        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    return pd.DataFrame(rows)

def notion_update(df: pd.DataFrame) -> None:
    if df.empty:
        return
    import time as _t

    # Include BuySellLots among numeric props
    number_cols = ["open", "high", "low", "close", "PrevClose", "DayGain", "Gain", "ATRP", "MARatio", "BuySellLots"]
    for c in number_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for _, r in df.iterrows():
        page_id = str(r["pageID"]).strip()
        props: Dict[str, Any] = {}

        # Numbers (avoid NaN/inf)
        for c in number_cols:
            val = r.get(c)
            if pd.notna(val):
                if isinstance(val, (int, float)) and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                    props[c] = {"number": float(val)}

        # Date
        if pd.notna(r.get("Date")):
            props["Date"] = {"date": {"start": str(r["Date"])}}

        # Selects
        if pd.notna(r.get("Status")):
            props["Status"] = {"select": {"name": str(r["Status"])}}
        if pd.notna(r.get("PrevStatus")):
            props["PrevStatus"] = {"select": {"name": str(r["PrevStatus"])}}

        if props:
            notion.pages.update(page_id=page_id, properties=props)
            _t.sleep(NOTION_SLEEP_SECONDS)  # ~3 req/sec

# ================== KITE / METRICS ==================
def status_check(df: pd.DataFrame) -> pd.DataFrame:
    status_condition = [
        (df["close"] > df["UpperRes"]),
        ((df["high"] > df["UpperRes"]) & (df["close"] < df["UpperRes"])),
        ((df["close"] > df["LowerRes"]) & (df["close"] < df["UpperRes"])),
        ((df["high"] > df["LowerRes"]) & (df["close"] < df["LowerRes"])),
        (df["close"] < df["LowerSup"]),
        ((df["low"] < df["LowerSup"]) & (df["close"] > df["LowerSup"])),
        ((df["close"] < df["UpperSup"]) & (df["close"] > df["LowerSup"])),
        ((df["high"] < df["UpperSup"]) & (df["close"] > df["UpperSup"])),
    ]
    statuses = [
        "Broke Resistance", "Tested Max Resistance", "In Resistance Range", "Tested Min Resistance",
        "Broke Support", "Tested Max Support", "In Support Range", "Tested Min Support",
    ]
    df["Status"] = np.select(status_condition, statuses, default="NA")

    prevstatus_condition = [
        (df["PrevClose"] > df["UpperRes"]),
        ((df["high"].shift(1) > df["UpperRes"]) & (df["PrevClose"] < df["UpperRes"])),
        ((df["PrevClose"] > df["LowerRes"]) & (df["PrevClose"] < df["UpperRes"])),
        ((df["high"].shift(1) > df["LowerRes"]) & (df["PrevClose"] < df["LowerRes"])),
        (df["PrevClose"] < df["LowerSup"]),
        ((df["low"].shift(1) < df["LowerSup"]) & (df["PrevClose"] > df["LowerSup"])),
        ((df["PrevClose"] < df["UpperSup"]) & (df["PrevClose"] > df["LowerSup"])),
        ((df["low"].shift(1) < df["UpperSup"]) & (df["PrevClose"] > df["UpperSup"])),
    ]
    df["PrevStatus"] = np.select(prevstatus_condition, statuses, default="NA")
    return df

def extract_stock_data(entoken: str, start_date: date, end_date: date, notion_input: pd.DataFrame, fund: float) -> pd.DataFrame:
    if notion_input.empty:
        return pd.DataFrame()

    kite = KiteApp(enctoken=entoken)
    frames: List[pd.DataFrame] = []

    for token in notion_input["instrument_token"].dropna().astype(str).tolist():
        data = kite.historical_data(instrument_token=token, from_date=start_date, to_date=end_date, interval="day")
        df = pd.DataFrame(data)
        if df.empty:
            continue

        df = df.sort_values("date")
        df["Date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["PrevClose"] = df["close"].shift(1)
        df["PrevHigh"] = df["high"].shift(1)
        df["PrevLow"] = df["low"].shift(1)
        df["DayGain"] = ((df["close"] - df["open"]) * 100 / df["open"]).round(2)
        df["Gain"] = ((df["close"] - df["PrevClose"]) * 100 / df["PrevClose"]).round(2)

        tr1 = (df["high"] - df["low"]).abs()
        tr2 = (df["high"] - df["PrevClose"]).abs()
        tr3 = (df["PrevClose"] - df["low"]).abs()
        df["ATRP"] = (
            pd.concat([tr1, tr2, tr3], axis=1)
            .max(axis=1)
            .ewm(span=5, adjust=True)
            .mean()
            .pipe(lambda s: (s * 100 / df["close"]).round(2))
        )

        df["instrument_token"] = token
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    final = pd.merge(notion_input, combined, on="instrument_token", how="inner")

    # Existing metric
    final["MARatio"] = (
        pd.to_numeric(final["MARGIN_RATE"], errors="coerce") /
        pd.to_numeric(final["ATRP"], errors="coerce")
    ).replace([np.inf, -np.inf], np.nan).round(2)

    # NEW logic: BuySellLots = floor(fund / MARGIN) * LOT_SIZE  (whole lots first)
    mgn = pd.to_numeric(final["MARGIN"], errors="coerce")
    lot = pd.to_numeric(final["LOT_SIZE"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        lots = np.floor(float(fund) / mgn)          # whole number of lots
        final["BuySellLots"] = (lots * lot)         # resulting quantity (shares)
        # Make it integral where possible, allow NA
        final["BuySellLots"] = pd.to_numeric(final["BuySellLots"], errors="coerce").round(0).astype("Int64")

    final = status_check(final)

    drop_cols = ["instrument_token", "date", "volume", "PrevHigh", "PrevLow"]
    final = final.drop(columns=[c for c in drop_cols if c in final.columns])

    if "Date" in final.columns:
        latest = final["Date"].max()
        final = final.loc[final["Date"] == latest]

    return final.reset_index(drop=True)

# ================== ROUTES ==================
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "FastAPI running"}

@app.post("/notion")
def run_notion(payload: Payload) -> Dict[str, str]:
    notion_input = fetch_notion_dataframe(payload.DATABASE_ID)
    start_date = payload.Date - timedelta(days=12)

    stock_data = extract_stock_data(
        entoken=payload.entoken.get_secret_value(),
        start_date=start_date,
        end_date=payload.Date,
        notion_input=notion_input,
        fund=payload.fund,
    )

    notion_update(stock_data)
    return {"message": "done"}