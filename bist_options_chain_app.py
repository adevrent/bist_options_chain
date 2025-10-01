# bist_options_chain_app.py
import io
from datetime import datetime
import pandas as pd
import streamlit as st

from bist_options_chain_functions import (
    load_r_array,
    get_business_days,
    create_multi_date_options_chain,
)

# --- App config ---
st.set_page_config(page_title="BIST Options Chain", layout="wide")

st.title("BIST Options Chain Analysis")
st.caption("Veri klasörünüz hazırsa bu arayüzle hızlıca zinciri çıkarıp Excel'e aktarabilirsiniz.")

# --- Inputs (exactly 3 fields) ---
col1, col2, col3 = st.columns(3)

stock_code = col1.text_input(
    "Hisse Kodu",
    value="",
    placeholder="e.g. ASELS",
)

start_date_ISO = col2.text_input(
    "Tarih Başlangıç",
    value="",
    placeholder="e.g. 2025-08-04",
)

end_date_ISO = col3.text_input(
    "Tarih Bitiş",
    value="",
    placeholder="e.g. 2025-08-11",
)

# --- Cached TLREF loader (reads once) ---
R_FILEPATH = "data/TLREFORANI_D.csv"

@st.cache_data(show_spinner=False)
def load_rates(path: str) -> pd.DataFrame:
    return load_r_array(path)  # returns a DataFrame indexed by Date with 'r_cont'

@st.cache_data(show_spinner=False)
def build_chain(dates_ISO, stock_code: str, r_df: pd.DataFrame) -> pd.DataFrame:
    return create_multi_date_options_chain(dates_ISO, stock_code, r_df)

# --- Action button ---
run = st.button("Verileri Getir")

def _validate_dates(s: str, e: str):
    """Validate YYYY-MM-DD strings and return them (ordered)."""
    try:
        sd = datetime.strptime(s.strip(), "%Y-%m-%d").date()
        ed = datetime.strptime(e.strip(), "%Y-%m-%d").date()
    except Exception:
        raise ValueError("Tarih formatı YYYY-MM-DD olmalıdır (örn. 2025-08-04).")
    if ed < sd:
        # swap if user reversed
        sd, ed = ed, sd
    return sd.isoformat(), ed.isoformat()

if run:
    # Basic input checks
    if not stock_code:
        st.error("Lütfen **Hisse Kodu** girin (örn. ASELS).")
        st.stop()
    try:
        start_date_ISO, end_date_ISO = _validate_dates(start_date_ISO, end_date_ISO)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    with st.spinner("Veriler çekiliyor ve zincir oluşturuluyor..."):
        # Load rates once
        r_array = load_rates(R_FILEPATH)

        # Business days in range
        dates_ISO = get_business_days(start_date_ISO, end_date_ISO)

        if not dates_ISO:
            st.warning("Seçilen aralıkta iş günü bulunamadı.")
            st.stop()

        # Ensure rates cover requested dates
        missing = [d for d in dates_ISO if d not in r_array.index.strftime("%Y-%m-%d")]
        if missing:
            st.warning(
                "Aşağıdaki tarihler için TLREF verisi bulunamadı ve bu günler atlanacak:\n\n"
                + ", ".join(missing)
            )
            dates_ISO = [d for d in dates_ISO if d not in missing]
            if not dates_ISO:
                st.error("Geçerli TLREF verisi olan gün kalmadı.")
                st.stop()

        # Build the full options chain DataFrame
        try:
            df_all = build_chain(dates_ISO, stock_code.upper(), r_array)
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success(
        f"Tamamlandı • {stock_code.upper()} • {start_date_ISO} → {end_date_ISO} • {len(df_all):,} satır"
    )

    # Show full DataFrame
    st.dataframe(df_all, use_container_width=True)

    # Export to Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_all.to_excel(writer, sheet_name="OptionsChain")
        # Autosize columns a bit
        worksheet = writer.sheets["OptionsChain"]
        for idx, col in enumerate(df_all.reset_index().columns):
            # set a reasonable width based on header + data
            width = max(
                len(str(col)),
                min(50, int(df_all.reset_index()[col].astype(str).str.len().mean() + 6)),
            )
            worksheet.set_column(idx, idx, width)

    st.download_button(
        label="Excel'e Aktar",
        data=buffer.getvalue(),
        file_name=f"{stock_code.upper()}_{start_date_ISO}_{end_date_ISO}_options_chain.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
