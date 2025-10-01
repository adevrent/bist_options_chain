from bist_options_chain_functions import load_r_array, create_multi_date_options_chain, get_business_days

r_filepath = r"data/TLREFORANI_D.csv"
r_array = load_r_array(r_filepath)

stock_code = "THYAO"
start_date_ISO = "2025-01-21"
end_date_ISO = "2025-01-31"
dates_ISO = get_business_days(start_date_ISO, end_date_ISO)

df_all = create_multi_date_options_chain(dates_ISO, stock_code, r_array)
print(df_all)