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


def get_asset_price_on_dates(dates_ISO, stock_code):
    start_date_ISO = dates_ISO[0]
    end_date_ISO = dates_ISO[-1]
    start_date_dt = convert_datetype(dates_ISO[0], "datetime")
    end_date_dt = convert_datetype(dates_ISO[-1], "datetime")
    df = yf.download(
        stock_code + ".IS",
        start=start_date_dt,
        end=end_date_dt + dt.timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"yfinance returned no data for {stock_code}.IS on {date_ISO}")
    return df.loc[:, "Close"]

def get_maturity_date_ISO_bday(maturity_code, calendar=ql.Turkey()):
    month = int(maturity_code[:2])
    year = int("20" + maturity_code[2:])
    day = 1
    first_day_of_month_ISO = convert_datetype(ql.Date(day, month, year), "ISO")
    last_bday_QL = get_last_business_day_of_month(first_day_of_month_ISO)
    maturity_date_ISO_bday = convert_datetype(last_bday_QL, "ISO")
    return maturity_date_ISO_bday

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


def get_asset_multi_day_options_chain(dates_ISO, stock_code, derivative_type="O", calendar=ql.Turkey()):
    S_array = get_asset_price_on_dates(dates_ISO, stock_code)
    # print(S_array)
    df_asset_all_dates = pd.DataFrame()
    for date_ISO in dates_ISO:
        year, month, day = date_ISO.split("-")

        filename = "VIOP_GUNSONU_FIYATHACIM.M." + year + month + ".csv"
        filepath = f"data\\{filename}"

        df_all_assets = pd.read_csv(filepath, header=1, sep=";")
        df_all_assets.set_index("TRADE DATE", inplace=True)

        if date_ISO not in df_all_assets.index:
            raise ValueError(f"No data for {date_ISO} in {filename}")

        day_slice = df_all_assets.loc[date_ISO, :]
        if isinstance(day_slice, pd.Series):
            df_all_assets_day = day_slice.to_frame().T
        else:
            df_all_assets_day = day_slice.copy()

        # filter to the underlying & options only
        df_asset = df_all_assets_day[
            (df_all_assets_day["UNDERLYING"] == stock_code + ".E") &
            (df_all_assets_day["INSTRUMENT SERIES"].str[0] == derivative_type)
        ].copy()

        # only with positive traded value
        if "TRADED VALUE" not in df_asset.columns:
            raise ValueError(f"'TRADED VALUE' column not found in data for {date_ISO}")

        df_asset = df_asset[df_asset["TRADED VALUE"] > 0]
        if df_asset.empty:
            print(f"No options with positive traded value for {stock_code} on {date_ISO}")
            continue

        df_asset = df_asset.loc[:, ["INSTRUMENT SERIES", "CLOSING PRICE", "TRADED VALUE"]]
        df_asset.columns = ["Contract", "Close Price", "Volume"]

        pattern = r'^[A-Z]_([A-Z]+?)E(\d{4})([CP])([\d.]+)$'
        df_asset[["Ticker", "Maturity Code", "Option Type", "Strike"]] = df_asset["Contract"].str.extract(pattern)
        df_asset.loc[:, "Maturity Date"] = None  # to be filled

        df_asset = df_asset.dropna(subset=["Maturity Code", "Option Type", "Strike"])
        if df_asset.empty:
            raise ValueError(f"No valid options data after extraction for {stock_code} on {date_ISO}")

        maturity_date_ISO_bday_array = []

        if type(df_asset.loc[date_ISO, "Maturity Code"]) == str:
            maturity_code = df_asset.loc[date_ISO, "Maturity Code"]
            maturity_date_ISO_bday = get_maturity_date_ISO_bday(maturity_code, calendar)
            maturity_date_ISO_bday_array.append(maturity_date_ISO_bday)

        else:
            for contract, maturity_code in zip(df_asset.loc[date_ISO, "Contract"].values, df_asset.loc[date_ISO, "Maturity Code"].values):
                maturity_date_ISO_bday = get_maturity_date_ISO_bday(maturity_code, calendar)
                maturity_date_ISO_bday_array.append(maturity_date_ISO_bday)

        if len(maturity_date_ISO_bday_array) == 1:
            df_asset.loc[date_ISO, "Maturity Date"] = convert_datetype(maturity_date_ISO_bday_array[0], "datetime")
        else:
            df_asset.loc[date_ISO, "Maturity Date"] = pd.to_datetime(maturity_date_ISO_bday_array).date

        try:
            S = S_array.loc[date_ISO].values[0]
            df_asset.loc[date_ISO, "Spot Price"] = S
        except KeyError as e:
            print("Error assigning Spot Price:", e)
            print("S:", S)
            print("df_asset shape:", df_asset.shape)

        df_asset["Strike"] = df_asset["Strike"].astype(float)

        # print(df_asset.dtypes)

        # df_asset = df_asset.sort_values(["Maturity Date", "Strike"])
        df_asset_all_dates = pd.concat([df_asset_all_dates, df_asset])
        # print(df_asset_all_dates)
    return df_asset_all_dates

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
        return np.nan
        return 0.001


def calc_iv_for_options_chain(df_asset_all_dates, r_array):
    iv_dict = {}
    df_asset_all_dates_iv = df_asset_all_dates.copy()

    for trade_date, row in df_asset_all_dates_iv.iterrows():
        # print(trade_date)
        # print(row)
        # print("Today Date:", row[0])
        # print("Maturity Date:", row[-1]["Maturity Date"])
        maturity_date = row.loc["Maturity Date"]
        r = r_array.loc[trade_date, "r_cont"]
        v = row.loc["Close Price"]
        phi = 1 if row.loc["Option Type"] == "C" else -1
        K = row.loc["Strike"]
        trade_QL = convert_datetype(trade_date, "QL")
        maturity_QL = convert_datetype(maturity_date, "QL")
        tau = ql.Actual365Fixed().yearFraction(trade_QL, maturity_QL)
        S = row.loc["Spot Price"]

        iv = get_iv_from_price(v, S, K, r, phi, tau)
        if trade_date not in iv_dict:
            iv_dict[trade_date] = []
        iv_dict[trade_date].append(iv)

    df_asset_all_dates_iv.loc[:, "Implied Volatility"] = np.nan
    for trade_date, iv_list in iv_dict.items():
        df_asset_all_dates_iv.loc[trade_date, "Implied Volatility"] = iv_list

    return df_asset_all_dates_iv

def create_options_chain_with_iv(df_asset, r):  # wrapper function
    df_asset_iv = calc_iv_for_options_chain(df_asset, r)
    return df_asset_iv


def get_business_days(start_date_ISO, end_date_ISO, calendar=ql.Turkey()):
    start_date_QL = convert_datetype(start_date_ISO, "QL")
    end_date_QL = convert_datetype(end_date_ISO, "QL")
    # print("end_date_QL:", end_date_QL)
    bday_QL_array = []
    current_date_QL = start_date_QL
    while current_date_QL <= end_date_QL:
        if calendar.isBusinessDay(current_date_QL):
            bday_QL_array.append(current_date_QL)
        current_date_QL = current_date_QL + ql.Period(1, ql.Days)
    bday_ISO_array = [convert_datetype(d, "ISO") for d in bday_QL_array]
    return bday_ISO_array


# r_filepath = r"data\TLREFORANI_D.csv"
# r_array = load_r_array(r_filepath)
# # print(r_array)

# stock_code = "ASELS"
# start_date_ISO = "2024-11-10"
# end_date_ISO = "2025-08-30"
# dates_ISO = get_business_days(start_date_ISO, end_date_ISO)
# # print(dates_ISO)

# df_asset_all_dates = get_asset_multi_day_options_chain(
#     dates_ISO,
#     stock_code,
#     derivative_type="O",
#     calendar=ql.Turkey()
# )

# df_asset_all_dates_iv = calc_iv_for_options_chain(df_asset_all_dates, r_array)
# print(df_asset_all_dates_iv)
