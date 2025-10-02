# bist_options_chain_app.py
import io
from datetime import datetime
import pandas as pd
import streamlit as st
import QuantLib as ql
import re

# --- Import updated helpers from your functions module ---
from bist_options_chain_functions import (
    load_r_array,                    # unchanged
    get_business_days,               # unchanged
    calc_iv_for_options_chain,       # signature changed: (df_all_dates, r_array)
    convert_datetype,                # unchanged (expanded types)
    get_last_business_day_of_month,  # unchanged
    get_maturity_date_ISO_bday,      # NEW helper to compute maturity EOM bday
    get_asset_price_on_dates,        # REPLACES get_asset_price_on_date
)

# ---------------- App config ----------------
st.set_page_config(page_title="VIOP Options Chain Tarihsel Verileri, Atakan Devrent", layout="wide")
st.title("BIST VIOP Options Chain Tarihsel Verileri, Atakan Devrent")
st.caption("Veriler https://datastore.borsaistanbul.com/ den alınmıştır.")

# ---------------- Secrets / Config ----------------
DATA_BASE_URL = st.secrets.get("DATA_BASE_URL", "").rstrip("/")
TLREF_FILENAME = st.secrets.get("TLREF_FILENAME", "TLREFORANI_D.csv")

if not DATA_BASE_URL:
    st.error(
        "DATA_BASE_URL ayarlı değil. Lütfen Secrets içine `DATA_BASE_URL` ekleyin "
        "(örn. GitHub raw klasörü: https://raw.githubusercontent.com/<user>/<repo>/<branch>/data)"
    )
    st.stop()

# ---------------- UI ----------------
col1, col2, col3 = st.columns(3)
stock_code = col1.text_input("Hisse Kodu", value="", placeholder="örn. ASELS")
start_date_ISO = col2.text_input("Tarih Başlangıç", value="", placeholder="örn. 2025-08-04")
end_date_ISO   = col3.text_input("Tarih Bitiş", value="", placeholder="örn. 2025-08-11")
run = st.button("Verileri Getir")

# ---------------- Remote paths ----------------
def viop_month_url(date_ISO: str) -> str:
    y, m, _ = date_ISO.split("-")
    return f"{DATA_BASE_URL}/VIOP_GUNSONU_FIYATHACIM.M.{y}{m}.csv"

def tlref_url() -> str:
    return f"{DATA_BASE_URL}/{TLREF_FILENAME}"

# ---------------- Cached readers ----------------
@st.cache_data(show_spinner=False)
def load_rates_online(url: str) -> pd.DataFrame:
    # uses your load_r_array (works with HTTP URL)
    return load_r_array(url)

@st.cache_data(show_spinner=False)
def read_viop_month_csv(url: str) -> pd.DataFrame:
    # Official BIST monthly files: semicolon separator, data header on row index 1
    return pd.read_csv(url, header=1, sep=";")

def _validate_dates(s: str, e: str):
    try:
        sd = datetime.strptime(s.strip(), "%Y-%m-%d").date()
        ed = datetime.strptime(e.strip(), "%Y-%m-%d").date()
    except Exception:
        raise ValueError("Tarih formatı YYYY-MM-DD olmalıdır (örn. 2025-08-04).")
    if ed < sd:
        sd, ed = ed, sd
    return sd.isoformat(), ed.isoformat()

def _read_viop_daily_from_month_csv(date_ISO: str) -> pd.DataFrame:
    """
    Read the monthly VIOP file for the given date from remote, filter to that day,
    and return the per-asset rows. Index remains 'TRADE DATE' (strings like YYYY-MM-DD).
    """
    url = viop_month_url(date_ISO)
    df_all_assets = read_viop_month_csv(url)
    df_all_assets.set_index("TRADE DATE", inplace=True)

    if date_ISO not in df_all_assets.index:
        raise FileNotFoundError(f"{date_ISO} tarihine ait satır bulunamadı. URL: {url}")

    day_slice = df_all_assets.loc[date_ISO, :]
    if isinstance(day_slice, pd.Series):
        return day_slice.to_frame().T.copy()
    return day_slice.copy()

def _parse_contract_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    From 'INSTRUMENT SERIES' extract ticker/maturity/type/strike exactly as in your functions.
    """
    df = df.copy()
    df = df.loc[:, ["INSTRUMENT SERIES", "CLOSING PRICE", "TRADED VALUE"]]
    df.columns = ["Contract", "Close Price", "Volume"]

    pattern = r'^[A-Z]_([A-Z]+?)E(\d{4})([CP])([\d.]+)$'
    # extract into 4 new columns
    df[["Ticker", "Maturity Code", "Option Type", "Strike"]] = df["Contract"].str.extract(pattern)
    df = df.dropna(subset=["Maturity Code", "Option Type", "Strike"])
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    return df

def _add_maturity_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the new get_maturity_date_ISO_bday helper to generate 'Maturity Date' (datetime.date).
    """
    df = df.copy()
    # vectorized apply through Python-level apply is fine (small-ish per-day dataset)
    df["Maturity Date"] = df["Maturity Code"].apply(get_maturity_date_ISO_bday)
    # convert ISO->datetime.date for compatibility with convert_datetype in IV function
    df["Maturity Date"] = df["Maturity Date"].apply(lambda d: convert_datetype(d, "datetime"))
    return df

def get_asset_options_chain_online(date_ISO: str, stock_code: str, derivative_type: str = "O") -> pd.DataFrame:
    """
    Online analogue of your multi-day reader, but for a single trade date.
    Filters UNDERLYING == '<CODE>.E' and instrument series starting with 'O'.
    Adds parsed contract fields and maturity date; 'Spot Price' is set outside.
    """
    df_day = _read_viop_daily_from_month_csv(date_ISO)

    # filter to underlying + options
    df_asset = df_day[
        (df_day["UNDERLYING"] == f"{stock_code}.E") &
        (df_day["INSTRUMENT SERIES"].str[0] == derivative_type)
    ].copy()

    # only with positive traded value
    if "TRADED VALUE" not in df_asset.columns:
        raise ValueError(f"'TRADED VALUE' column yok: {date_ISO}")
    df_asset = df_asset[df_asset["TRADED VALUE"] > 0]
    if df_asset.empty:
        return df_asset  # empty for this day

    df_asset = _parse_contract_columns(df_asset)
    df_asset = _add_maturity_date(df_asset)
    return df_asset

@st.cache_data(show_spinner=False)
def get_spot_price_map(dates_ISO, stock_code: str) -> dict:
    """
    Fetches all spot prices once using the new get_asset_price_on_dates,
    and returns a { 'YYYY-MM-DD': close } dict for quick assignment.
    """
    s = get_asset_price_on_dates(dates_ISO, stock_code)  # Series of Close prices
    # normalize index to ISO strings (to align with TRADE DATE index)
    s.index = pd.to_datetime(s.index).strftime("%Y-%m-%d")
    return s.to_dict()

@st.cache_data(show_spinner=False)
def build_multi_date_options_chain_online(dates_ISO, stock_code: str) -> pd.DataFrame:
    """
    Build a concatenated options chain for all dates; the index remains the trade date strings.
    Spot prices are assigned from a pre-fetched map (one yfinance call).
    """
    dfs = []
    spot_map = get_spot_price_map(dates_ISO, stock_code)

    for d in dates_ISO:
        df_d = get_asset_options_chain_online(d, stock_code)
        if df_d is None or df_d.empty:
            continue
        # assign spot for *every* row of that day
        spot = spot_map.get(d, None)
        if spot is not None:
            df_d.loc[:, "Spot Price"] = float(spot)
        else:
            # if spot is missing, set a tiny positive value to avoid log issues
            df_d.loc[:, "Spot Price"] = 1e-6
        dfs.append(df_d)

    if not dfs:
        return pd.DataFrame()

    # concat keeps the original index (TRADE DATE), which is required by calc_iv_for_options_chain
    return pd.concat(dfs, axis=0)

# ---------------- RUN ----------------
if run:
    if not stock_code:
        st.error("Lütfen **Hisse Kodu** girin (örn. ASELS).")
        st.stop()
    try:
        start_date_ISO, end_date_ISO = _validate_dates(start_date_ISO, end_date_ISO)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    with st.spinner("TLREF yükleniyor..."):
        r_array = load_rates_online(tlref_url())

    dates_ISO_all = get_business_days(start_date_ISO, end_date_ISO)
    if not dates_ISO_all:
        st.warning("Seçilen aralıkta iş günü yok.")
        st.stop()

    # ensure TLREF coverage (r_array index is DatetimeIndex)
    tlref_dates = set(pd.to_datetime(r_array.index).strftime("%Y-%m-%d"))
    usable = [d for d in dates_ISO_all if d in tlref_dates]
    missing = [d for d in dates_ISO_all if d not in tlref_dates]
    if missing:
        st.info("TLREF olmayan günler atlandı: " + ", ".join(missing))
    if not usable:
        st.error("Seçilen aralıkta TLREF olan gün bulunamadı.")
        st.stop()

    with st.spinner("Opsiyon zinciri indiriliyor ve hazırlanıyor..."):
        df_all = build_multi_date_options_chain_online(usable, stock_code.upper())

    if df_all.empty:
        st.warning("Seçilen aralıkta ilgili enstrüman için işlem gören opsiyon bulunamadı.")
        st.stop()

    with st.spinner("Zımni oynaklık hesaplanıyor..."):
        # NEW signature: pass entire r_array (not a scalar r)
        df_all_iv = calc_iv_for_options_chain(df_all, r_array)

    st.success(
        f"Tamamlandı • {stock_code.upper()} • {start_date_ISO} → {end_date_ISO} • {len(df_all_iv):,} satır"
    )

    st.dataframe(df_all_iv, use_container_width=True)

    # Export to Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_all_iv.to_excel(writer, sheet_name="OptionsChain")
        ws = writer.sheets["OptionsChain"]
        df_reset = df_all_iv.reset_index()
        for idx, col in enumerate(df_reset.columns):
            width = max(len(str(col)), min(50, int(df_reset[col].astype(str).str.len().mean() + 6)))
            ws.set_column(idx, idx, width)

    st.download_button(
        label="Excel'e Aktar",
        data=buffer.getvalue(),
        file_name=f"{stock_code.upper()}_{start_date_ISO}_{end_date_ISO}_options_chain.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
