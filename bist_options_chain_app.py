# bist_options_chain_app.py
import io
from datetime import datetime
import pandas as pd
import streamlit as st

# ---- your functions (unchanged calcs) ----
from functions import (
    load_r_array,
    get_business_days,
    get_asset_multi_day_options_chain,
    calc_iv_for_options_chain,
)

# ---------------- App config ----------------
st.set_page_config(page_title="BIST VIOP Options Chain Tarihsel Verileri, Atakan Devrent", layout="wide")
st.title("BIST VIOP Options Chain Tarihsel Verileri, Atakan Devrent")
st.caption("Veriler https://datastore.borsaistanbul.com/ den alınmıştır. 2024-11-01 ve 2025-08-30 arası tarihler uygundur. Eylül ayı datasını BIST henüz paylaşmamıştır,.")

# ---------------- Inputs ----------------
c1, c2, c3 = st.columns(3)
stock_code = c1.text_input("Hisse Kodu", value="", placeholder="örn. ASELS").strip().upper()
start_date_raw = c2.text_input("Tarih Başlangıç", value="", placeholder="örn. 2025-08-04")
end_date_raw   = c3.text_input("Tarih Bitiş", value="", placeholder="örn. 2025-08-30-11")

run = st.button("Verileri Getir", use_container_width=True)

# ---------------- Helpers ----------------
def _clean_iso(d: str) -> str:
    """
    Accepts strings like '2025-08-30' or '2025-08-30-11'.
    Returns the first 10 chars (YYYY-MM-DD) if valid, otherwise raises.
    """
    d = (d or "").strip()
    if len(d) >= 10:
        d = d[:10]
    # simple validation but do not alter your calculations
    datetime.strptime(d, "%Y-%m-%d")
    return d

def _build_tlref_url() -> str:
    """
    r_filepath = 'data/TLREFORANI_D.csv' under your repo.
    Load via DATA_BASE_URL from Streamlit secrets.
    """
    base = st.secrets.get("DATA_BASE_URL", "").rstrip("/")
    if not base:
        st.error("Secrets içinde DATA_BASE_URL bulunamadı.")
        st.stop()
    r_filepath = "TLREFORANI_D.csv"
    return f"{base}/{r_filepath}"

# ---------------- Run ----------------
if run:
    if not stock_code:
        st.error("Lütfen **Hisse Kodu** girin (örn. ASELS).")
        st.stop()
    try:
        start_date_ISO = _clean_iso(start_date_raw)
        end_date_ISO   = _clean_iso(end_date_raw)
    except Exception:
        st.error("Tarih formatı `YYYY-MM-DD` olmalıdır (örn. 2025-08-04).")
        st.stop()

    # 1) TLREF'i URL'den yükle (aynı şekilde yerelinizdeki load_r_array çağrısı)
    with st.spinner("TLREF verisi yükleniyor..."):
        tlref_url = _build_tlref_url()
        r_array = load_r_array(tlref_url)

    # 2) İş günleri aralığı
    with st.spinner("İş günleri hesaplanıyor..."):
        dates_ISO = get_business_days(start_date_ISO, end_date_ISO)

    # 3) Çok-günlü opsiyon zinciri
    with st.spinner("Opsiyon zinciri indiriliyor..."):
        df_asset_all_dates = get_asset_multi_day_options_chain(
            dates_ISO,
            stock_code,
            derivative_type="O",
        )

    if df_asset_all_dates is None or df_asset_all_dates.empty:
        st.warning("Seçilen aralıkta veri bulunamadı.")
        st.stop()

    # 4) IV
    with st.spinner("Implied Vol hesaplanıyor..."):
        df_iv = calc_iv_for_options_chain(df_asset_all_dates, r_array)

    st.success(f"Tamamlandı • {stock_code} • {start_date_ISO} → {end_date_ISO} • {len(df_iv):,} satır")
    st.dataframe(df_iv, use_container_width=True)

    # 5) Excel'e aktar
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_iv.to_excel(writer, sheet_name="OptionsChain")
        # Kolon genişliklerini kabaca ayarla
        ws = writer.sheets["OptionsChain"]
        df_reset = df_iv.reset_index()
        for idx, col in enumerate(df_reset.columns):
            width = max(len(str(col)), min(50, int(df_reset[col].astype(str).str.len().mean() + 6)))
            ws.set_column(idx, idx, width)

    st.download_button(
        label="Excel'e aktar",
        data=buffer.getvalue(),
        file_name=f"{stock_code}_{start_date_ISO}_{end_date_ISO}_options_chain.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
