# bist_options_chain_app.py
import io
from datetime import datetime
import pandas as pd
import streamlit as st

# Reuse your existing logic/helpers
from bist_options_chain_functions import (
    load_r_array,                 # works with URLs too
    get_business_days,
    calc_iv_for_options_chain,
    convert_datetype,
    get_last_business_day_of_month,
    get_asset_price_on_date,
)
import QuantLib as ql
import re

# ---------------- App config ----------------
st.set_page_config(page_title="VIOP Options Chain Tarihsel Verileri, Atakan Devrent", layout="wide")
st.title("BIST VIOP Options Chain Tarihsel Verileri, Atakan Devrent")
st.caption("Veriler https://datastore.borsaistanbul.com/ adresinden alınmıştır. (Eylül verileri henüz yayınlanmadığı için 2024-08-30 sonrası tarihlerde veri yoktur.)")

# ---------------- Secrets / Config ----------------
DATA_BASE_URL = st.secrets.get("DATA_BASE_URL", "").rstrip("/")
TLREF_FILENAME = st.secrets.get("TLREF_FILENAME", "TLREFORANI_D.csv")

if not DATA_BASE_URL:
    st.error(
        "DATA_BASE_URL ayarlı değil. Lütfen Secrets içine `DATA_BASE_URL` ekleyin "
        "(örn. GitHub raw klasörü: https://raw.githubusercontent.com/<user>/<repo>/<branch>/data)"
    )
    st.stop()

# ---------------- UI: exactly 3 fields ----------------
col1, col2, col3 = st.columns(3)

stock_code = col1.text_input("Hisse Kodu", value="", placeholder="e.g. ASELS")
start_date_ISO = col2.text_input("Tarih Başlangıç", value="", placeholder="e.g. 2025-08-04")
end_date_ISO   = col3.text_input("Tarih Bitiş", value="", placeholder="e.g. 2025-08-11")

run = st.button("Verileri Getir")

# ---------------- Remote readers ----------------
def viop_month_url(date_ISO: str) -> str:
    y, m, _ = date_ISO.split("-")
    return f"{DATA_BASE_URL}/VIOP_GUNSONU_FIYATHACIM.M.{y}{m}.csv"

def tlref_url() -> str:
    return f"{DATA_BASE_URL}/{TLREF_FILENAME}"

@st.cache_data(show_spinner=False)
def load_rates_online(url: str) -> pd.DataFrame:
    # Your loader uses pd.read_csv with utf-16 and ';' — it works with HTTP URLs too.
    return load_r_array(url)  # returns DF indexed by datetime with r_cont. :contentReference[oaicite:2]{index=2}

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
    and return the per-asset rows (header row is 1; ';' separator) just like your local version. :contentReference[oaicite:3]{index=3}
    """
    url = viop_month_url(date_ISO)
    # The official files use ';' and headers start at row index 1 (skip header line 0)
    df_all_assets = pd.read_csv(url, header=1, sep=";")
    # 'TRADE DATE' exists in your code and is used as index; then filtered by ISO date. :contentReference[oaicite:4]{index=4}
    df_all_assets.set_index("TRADE DATE", inplace=True)
    if date_ISO not in df_all_assets.index:
        raise FileNotFoundError(
            f"{date_ISO} tarihine ait satır bulunamadı. URL: {url}"
        )
    return df_all_assets.loc[date_ISO, :].copy()

def get_asset_options_chain_online(date_ISO: str, stock_code: str, derivative_type: str = "O") -> pd.DataFrame:
    """
    Online clone of your get_asset_options_chain: builds the same DataFrame but reads from HTTP instead of local disk.
    Mirrors: filter by UNDERLYING, INSTRUMENT SERIES[0] == 'O', parse fields via regex, add maturity date, spot, etc. :contentReference[oaicite:5]{index=5}
    """
    df_all_assets_day = _read_viop_daily_from_month_csv(date_ISO)

    # same filtering as your code
    df_asset = df_all_assets_day[
        (stock_code + ".E" == df_all_assets_day.loc[:, "UNDERLYING"]) &
        (df_all_assets_day.loc[:, "INSTRUMENT SERIES"].str[0] == derivative_type)
    ].copy()

    # only with positive traded value
    df_asset = df_asset[df_asset.loc[:, "TRADED VALUE"] > 0]
    df_asset = df_asset.loc[:, ["INSTRUMENT SERIES", "CLOSING PRICE", "TRADED VALUE"]]
    df_asset.columns = ["Contract", "Close Price", "Volume"]

    # parse series into columns as in your regex
    pattern = r'^[A-Z]_([A-Z]+?)E(\d{4})([CP])([\d.]+)$'
    df_asset[["Ticker", "Maturity Code", "Option Type", "Strike"]] = df_asset["Contract"].str.extract(pattern)

    # maturity date (last business day of maturity month)
    maturity_date_ISO_bday_array = []
    for maturity_code in df_asset["Maturity Code"].values:
        # MMYY -> month, year
        month = int(maturity_code[:2])
        year  = int("20" + maturity_code[2:])
        day = 1
        # last business day in TR calendar
        maturity_date_ISO = convert_datetype(ql.Date(day, month, year), "ISO")
        last_bday_QL = get_last_business_day_of_month(maturity_date_ISO)
        maturity_date_ISO_bday = convert_datetype(last_bday_QL, "ISO")
        maturity_date_ISO_bday_array.append(maturity_date_ISO_bday)

    df_asset.loc[:, "Maturity Date"] = pd.to_datetime(maturity_date_ISO_bday_array).date

    # spot from yfinance (you already do this) :contentReference[oaicite:6]{index=6}
    S = get_asset_price_on_date(date_ISO, stock_code)
    df_asset.loc[:, "Spot Price"] = S

    # cleanup
    df_asset["Strike"] = df_asset["Strike"].astype(float)
    df_asset = df_asset.sort_values(["Maturity Date", "Strike"])
    return df_asset

def create_options_chain_online(date_ISO: str, stock_code: str, r: float) -> pd.DataFrame:
    df_asset = get_asset_options_chain_online(date_ISO, stock_code)
    return calc_iv_for_options_chain(df_asset, r)

@st.cache_data(show_spinner=False)
def create_multi_date_options_chain_online(dates_ISO, stock_code: str, r_array: pd.DataFrame, add_next_bd: bool = False) -> pd.DataFrame:
    list_of_dfs = []
    tr_cal = ql.Turkey()

    for date_ISO in dates_ISO:
        r_today = r_array.loc[date_ISO, "r_cont"]
        df_today = create_options_chain_online(date_ISO, stock_code, r_today)
        df_today.loc[:, "Date"] = date_ISO
        list_of_dfs.append(df_today)

        if add_next_bd:
            next_date_QL = convert_datetype(date_ISO, "QL")
            next_date_QL = tr_cal.advance(next_date_QL, ql.Period(1, ql.Days))
            next_date_ISO = convert_datetype(next_date_QL, "ISO")

            r_next = r_array.loc[next_date_ISO, "r_cont"]
            df_next = create_options_chain_online(next_date_ISO, stock_code, r_next)
            df_next.loc[:, "Date"] = next_date_ISO
            list_of_dfs.append(df_next)

    df_all = pd.concat(list_of_dfs).drop(columns=["Date", "Maturity Code"])
    return df_all

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

    dates_ISO = get_business_days(start_date_ISO, end_date_ISO)
    if not dates_ISO:
        st.warning("Seçilen aralıkta iş günü yok.")
        st.stop()

    # ensure TLREF coverage
    tlref_dates = set(pd.to_datetime(r_array.index).strftime("%Y-%m-%d"))
    usable = [d for d in dates_ISO if d in tlref_dates]
    missing = [d for d in dates_ISO if d not in tlref_dates]
    if missing:
        st.info("TLREF olmayan günler atlandı: " + ", ".join(missing))
    if not usable:
        st.error("Seçilen aralıkta TLREF olan gün bulunamadı.")
        st.stop()

    with st.spinner("Opsiyon zinciri hesaplanıyor..."):
        df_all = create_multi_date_options_chain_online(usable, stock_code.upper(), r_array)

    st.success(
        f"Tamamlandı • {stock_code.upper()} • {start_date_ISO} → {end_date_ISO} • {len(df_all):,} satır"
    )

    # Show full DataFrame
    st.dataframe(df_all, use_container_width=True)

    # Export to Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_all.to_excel(writer, sheet_name="OptionsChain", index=False)
        workbook  = writer.book
        worksheet = writer.sheets["OptionsChain"]

        # autosize all columns
        for idx, col in enumerate(df_all.columns):
            width = max(len(str(col)),
                        min(50, int(df_all[col].astype(str).str.len().mean() + 6)))
            worksheet.set_column(idx, idx, width)

        # format "Implied Vol" column as percentage with 2 decimals
        if "Implied Vol" in df_all.columns:
            iv_col_idx = df_all.columns.get_loc("Implied Vol")
            percent_fmt = workbook.add_format({"num_format": "0.00%"})
            # just reuse the width you computed above
            worksheet.set_column(iv_col_idx, iv_col_idx, None, percent_fmt)

    st.download_button(
        label="Excel'e Aktar",
        data=buffer.getvalue(),
        file_name=f"{stock_code.upper()}_{start_date_ISO}_{end_date_ISO}_options_chain.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
