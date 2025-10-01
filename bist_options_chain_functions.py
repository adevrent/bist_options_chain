import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import QuantLib as ql
import datetime as dt
import scipy.stats as ss
from scipy.optimize import root_scalar
import yfinance as yf

# dates_filepath = r"data\Karlilik Raporlama Tarihleri.xlsx"
# earnings_dates = pd.read_excel(dates_filepath)

def load_r_array(r_filepath):
    # 1) Read with correct encoding and delimiter
    df = pd.read_csv(r_filepath, encoding="utf-16", sep=";")

    # 2) Parse the date column (day-first) and drop non-data footer rows
    df["date"] = pd.to_datetime(df["TARIH/DATE"], dayfirst=True, errors="coerce")
    df = df[df["date"].notna()].copy()

    # 3) (Optional) add ISO formatted date and sort
    df["date_iso"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    # 4) (Optional) set the date as the index for time-series work
    df = df.set_index("date")

    r_array = df.loc[:, ["DEGER/VALUE"]]
    # r_array.index.name = "Date"
    r_array.columns = ["r"]
    r_array.loc[:, "Date"] = r_array.index.values

    r_array["r_simple"] = pd.to_numeric(r_array["r"], errors="coerce") / 100
    r_array["tau"] = (r_array["Date"].shift(-1) - r_array["Date"]).dt.days.astype("Int64") / 365
    r_array["r_cont"] = np.log(1 + r_array["r_simple"] * r_array["tau"]) / r_array["tau"]
    r_array = r_array.drop(columns=["r", "Date"]).dropna().copy()
    r_array.index.name = "Date"

    return r_array

valid_datetypes = ["ISO", "datetime", "QL"]

def get_asset_price_on_date(date_ISO, stock_code):
    start_date_dt = convert_datetype(date_ISO, "datetime")
    end_date_dt = start_date_dt + dt.timedelta(days=1)
    df = yf.download(
        stock_code + ".IS",
        start=start_date_dt,
        end=end_date_dt,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"yfinance returned no data for {stock_code}.IS on {date_ISO}")
    return float(df["Close"].iloc[0])

def convert_datetype(date, to_type):
    assert to_type in valid_datetypes, "Invalid to_date type"

    # If the input is already the target type, return it.
    if to_type == "ISO" and isinstance(date, str):
        return date
    elif to_type == "datetime" and isinstance(date, dt.date) and not isinstance(date, dt.datetime):
        return date
    elif to_type == "QL" and isinstance(date, ql.Date):
        return date

    if isinstance(date, str):  # date is ISO
        assert len(date) == 10, "Date is not in valid ISO format"
        if to_type == "datetime":
            return dt.datetime.strptime(date, "%Y-%m-%d").date()
        elif to_type == "QL":
            return ql.DateParser.parseISO(date)
    elif isinstance(date, dt.date) and not isinstance(date, dt.datetime):  # date is datetime.date
        if to_type == "ISO":
            return date.strftime("%Y-%m-%d")
        elif to_type == "QL":
            return ql.Date.from_date(date)
    elif isinstance(date, dt.datetime):  # if date is datetime.datetime, convert to date first
        # Convert to date before further processing
        date_only = date.date()
        return convert_datetype(date_only, to_type)
    elif isinstance(date, ql.Date):  # date is QuantLib Date
        if to_type == "ISO":
            return date.ISO()
        elif to_type == "datetime":
            return date.to_date()

    # If none of the above conditions match, raise an error.
    raise TypeError("Unsupported date type provided.")

def get_last_business_day_of_month(date_ISO, calendar=ql.Turkey()):
    date_QL = convert_datetype(date_ISO, "QL")
    eom = ql.Date.endOfMonth(date_QL)
    last_bday_QL = calendar.adjust(eom, ql.Preceding)
    return last_bday_QL

def get_asset_options_chain(date_ISO, stock_code, derivative_type="O", calendar=ql.Turkey()):
    year, month, day = date_ISO.split("-")

    date_QL = convert_datetype(date_ISO, "QL")

    filename = "VIOP_GUNSONU_FIYATHACIM.M." + year + month + ".csv"
    filepath = f"data\{filename}"

    df_all_assets = pd.read_csv(filepath, header=1, sep=";")
    df_all_assets.set_index("TRADE DATE", inplace=True)
    df_all_assets = df_all_assets.loc[date_ISO, :]

    df_asset = df_all_assets[(stock_code + str(".E") == df_all_assets.loc[:, "UNDERLYING"]) & (df_all_assets.loc[:, "INSTRUMENT SERIES"].str[0] == derivative_type)]
    df_asset = df_asset[df_asset.loc[:, "TRADED VALUE"] > 0]
    df_asset = df_asset.loc[:, ["INSTRUMENT SERIES", "CLOSING PRICE", "TRADED VALUE"]]
    df_asset.columns = ["Contract", "Close Price", "Volume"]
    
    if df_asset.empty:
        return pd.DataFrame()

    pattern = r'^[A-Z]_([A-Z]+?)E(\d{4})([CP])([\d.]+)$'
    df_asset[["Ticker", "Maturity Code", "Option Type", "Strike"]] = df_asset["Contract"].str.extract(pattern)
    df_asset.loc[:, "Maturity Date"] = None  # to be filled
    
    df_asset = df_asset.dropna(subset=["Maturity Code", "Option Type", "Strike"])
    if df_asset.empty:
        return pd.DataFrame()

    maturity_date_ISO_bday_array = []
    S = get_asset_price_on_date(date_ISO, stock_code)
    
    for contract, maturity_code in zip(df_asset.loc[:, "Contract"].values, df_asset.loc[:, "Maturity Code"].values):
        month = int(maturity_code[:2])
        year = int("20" + maturity_code[2:])
        day = 1
        maturity_date_ISO = convert_datetype(ql.Date(day, month, year), "ISO")
        last_bday_QL = get_last_business_day_of_month(maturity_date_ISO)
        maturity_date_ISO_bday = convert_datetype(last_bday_QL, "ISO")
        maturity_date_ISO_bday_array.append(maturity_date_ISO_bday)

    df_asset.loc[:, "Maturity Date"] = pd.to_datetime(maturity_date_ISO_bday_array).date
    
    try:
        df_asset.loc[:, "Spot Price"] = S
    except ValueError as e:
        print("Error assigning Spot Price:", e)
        print("S:", S)
        print("df_asset shape:", df_asset.shape)
        df_asset.loc[:, "Spot Price"] = 1e-6  # Assign a very small number to avoid issues

    df_asset["Strike"] = df_asset["Strike"].astype(float)
    df_asset = df_asset.sort_values(["Maturity Date", "Strike"])
    return df_asset

def calc_d1(f, K, sigma, tau):
    """
    Calculate d1 for Black-Scholes formula.

    Args:
        K (float): strike price
        sigma (float): volatility of the option, in 0.33 format.
    """
    d1 = (np.log(f/K) + 0.5*(sigma**2)*tau) / (sigma*np.sqrt(tau))
    return d1

def calc_d2(f, K, sigma, tau):
    """
    Calculate d2 for Black-Scholes formula.

    Args:
        K (float): strike price
        sigma (float): volatility of the option, in 0.33 format.
    """
    d2 = (np.log(f/K) - 0.5*(sigma**2)*tau) / (sigma*np.sqrt(tau))
    return d2

def BS(phi, S, K, r, sigma, tau):
    f = S * np.exp(r * tau)  # forward price

    d1 = calc_d1(f, K, sigma, tau)
    d2 = calc_d2(f, K, sigma, tau)

    v = phi * np.exp(-r * tau) * (S * ss.norm.cdf(phi * d1) - K * ss.norm.cdf(phi * d2))
    delta_S = phi * ss.norm.cdf(phi * d1)
    
    # print("phi:", phi, " S:", S, " K:", K, " r:", r, " sigma:", sigma, " tau:", tau)

    return {"v":v, "delta_S":delta_S}

def get_iv_from_price(v, S, K, r, phi, tau, eps=1e-6, max_iter=10000):
    def f(sigma):
        BS_v = BS(phi, S, K, r, sigma, tau)["v"]
        return BS_v - v
    
    a, b = 1e-6, 1.0
    
    try:
        res = root_scalar(f, method="brentq", bracket=[a, b], xtol=eps, maxiter=max_iter)
        return np.maximum(res.root, 1e-12)
    except ValueError as e:
        # print("Root finding failed for parameters:")
        # print("v:", v, " S:", S, " K:", K, " r:", r, " phi:", phi, " tau:", tau)
        # print(e)
        # sigma_array = np.linspace(a, b, 100)
        # f_array = np.vectorize(f)(sigma_array)
        # plt.plot(sigma_array, f_array)
        # plt.axhline(0, color='red', linestyle='--')
        return 0.001

def calc_iv_for_options_chain(df_asset, r, S=None):
    iv_array = []
    df_asset_iv = df_asset.copy()
    if S is None:
        S = df_asset_iv.loc[:, "Spot Price"].values[0]
        # print(S)
    for row in df_asset_iv.iterrows():
        # print("Today Date:", row[0])
        # print("Maturity Date:", row[-1]["Maturity Date"])
        v = row[-1]["Close Price"]
        phi = 1 if row[-1]["Option Type"] == "C" else -1
        K = row[-1]["Strike"]
        today_QL = convert_datetype(row[0], "QL")
        maturity_QL = convert_datetype(row[-1]["Maturity Date"], "QL")
        tau = ql.Actual365Fixed().yearFraction(today_QL, maturity_QL)

        iv = get_iv_from_price(v, S, K, r, phi, tau)
        iv_array.append(iv)

    df_asset_iv.loc[:, "Implied Vol"] = iv_array
    return df_asset_iv

def create_options_chain(date_ISO, stock_code, r):  # wrapper function
    df_asset = get_asset_options_chain(date_ISO, stock_code)
    df_asset_iv = calc_iv_for_options_chain(df_asset, r)
    return df_asset_iv

def create_multi_date_options_chain(dates_ISO, stock_code, r_array, add_next_bd=False, calendar=ql.Turkey()):
    list_of_dfs = []
    for date_ISO in dates_ISO:
        # Earnings date
        r_today = r_array.loc[date_ISO, "r_cont"]
        df_asset_iv_today = create_options_chain(date_ISO, stock_code, r_today)
        df_asset_iv_today.loc[:, "Date"] = date_ISO
        list_of_dfs.append(df_asset_iv_today)
        
        if add_next_bd:
            # Next business day
            next_date_QL = convert_datetype(date_ISO, "QL")
            next_date_QL = calendar.advance(next_date_QL, ql.Period(1, ql.Days))
            next_date_ISO = convert_datetype(next_date_QL, "ISO")
            
            r_next = r_array.loc[next_date_ISO, "r_cont"]
            df_asset_iv_next_day = create_options_chain(next_date_ISO, stock_code, r_next)
            df_asset_iv_next_day.loc[:, "Date"] = next_date_ISO
            list_of_dfs.append(df_asset_iv_next_day)

    df_all = pd.concat(list_of_dfs).drop(columns=["Date", "Maturity Code"])
    return df_all

def get_business_days(start_date_ISO, end_date_ISO, calendar=ql.Turkey()):
    start_date_QL = convert_datetype(start_date_ISO, "QL")
    end_date_QL = convert_datetype(end_date_ISO, "QL")
    bday_QL_array = []
    current_date_QL = start_date_QL
    while current_date_QL <= end_date_QL:
        if calendar.isBusinessDay(current_date_QL):
            bday_QL_array.append(current_date_QL)
        current_date_QL = current_date_QL + ql.Period(1, ql.Days)
    bday_ISO_array = [convert_datetype(d, "ISO") for d in bday_QL_array]
    return bday_ISO_array