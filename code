import pandas as pd
import numpy as np
import re

INPUT_FILE      = "iGS_Raw_Sales_2026.csv"
OUTPUT_FILE     = "iGS_Cleaned_Sales_2026.csv"
FX_NTD_TO_USD   = 1 / 32.0
UNIT_SOLD_CAP   = 10_000
VALID_STATUS    = {"Shipped", "Pending", "Cancelled", "Returned"}
STATUS_PRIORITY = {"Shipped": 3, "Returned": 2, "Cancelled": 1, "Pending": 0}

df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)
df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

original_count = len(df)
log = []

def drop_rows(df, mask, reason):
    n = int(mask.sum())
    log.append((reason, n))
    return df[~mask].copy()

df = drop_rows(df, ~df["Order_ID"].str.match(r'^ORD-\d{6}$'),
               "Order_ID invalid format (expected ORD-XXXXXX)")

df = drop_rows(df, ~df["Customer_ID"].str.match(r'^CUST-\d{4}$'),
               "Customer_ID invalid format (expected CUST-XXXX)")

df["Status"] = df["Status"].str.capitalize()
df = drop_rows(df, ~df["Status"].isin(VALID_STATUS),
               f"Status not in {VALID_STATUS}")

def parse_revenue(val):
    s = str(val).strip().upper().replace(",", "")
    is_ntd = "NTD" in s
    clean = re.sub(r'[A-Z$\s]', '', s)
    try:
        amount = float(clean)
    except ValueError:
        return np.nan, is_ntd, True
    if amount < 0:
        return np.nan, is_ntd, True
    return (amount * FX_NTD_TO_USD) if is_ntd else amount, is_ntd, False

parsed = df["Revenue"].apply(parse_revenue)
df["Revenue_USD"] = parsed.apply(lambda x: x[0])
df["Is_Taiwan"]   = parsed.apply(lambda x: x[1])
df = drop_rows(df, parsed.apply(lambda x: x[2]),
               "Revenue unparseable or negative")

def parse_units(val):
    try:
        f = float(str(val).replace(",", "").strip())
    except ValueError:
        return np.nan
    if not (f > 0 and np.isfinite(f) and f == int(f)):
        return np.nan
    return int(f)

df["Unit_Sold"] = df["Unit_Sold"].apply(parse_units)
df = drop_rows(df, df["Unit_Sold"].isna(),
               "Unit_Sold not a positive integer or missing")
df = drop_rows(df, df["Unit_Sold"] > UNIT_SOLD_CAP,
               f"Unit_Sold > {UNIT_SOLD_CAP:,} (phantom row / data entry error)")

df = drop_rows(df, df["Product_Category"].str.strip() == "",
               "Product_Category is blank")

df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"], format="%m/%d/%Y", errors="coerce")
df["Return_Date"]   = pd.to_datetime(df["Return_Date"],   format="%m/%d/%Y", errors="coerce")

df = drop_rows(df, df["Purchase_Date"].isna(),
               "Purchase_Date missing or unparseable")

today = pd.Timestamp.today().normalize()
df = drop_rows(df, df["Return_Date"].notna() & (df["Return_Date"] > today),
               "Return_Date is in the future")
df = drop_rows(df, df["Return_Date"].notna() & (df["Return_Date"] < df["Purchase_Date"]),
               "Return_Date is earlier than Purchase_Date")
df = drop_rows(df, (df["Status"] == "Returned") & df["Return_Date"].isna(),
               "Status=Returned but Return_Date is missing")

before_dedup = len(df)

df["_priority"] = df["Status"].map(STATUS_PRIORITY).fillna(-1)
df = (df
      .sort_values(["Order_ID", "_priority"], ascending=[True, False])
      .drop_duplicates(subset="Order_ID", keep="first")
      .drop(columns=["_priority"])
      )

log.append((
    "Order_ID duplicate — kept highest-priority Status, removed secondary row",
    before_dedup - len(df)
))

df["Unit_Sold"]   = df["Unit_Sold"].astype(int)
df["Revenue_USD"] = df["Revenue_USD"].round(2)

df_clean = df[[
    "Order_ID", "Customer_ID", "Product_Category",
    "Purchase_Date", "Return_Date",
    "Revenue_USD", "Unit_Sold", "Status", "Is_Taiwan"
]].reset_index(drop=True)

df_clean.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

print("=" * 65)
print("  iGS GlobalRetail — DATA CLEANING REPORT")
print("=" * 65)
print(f"  Original rows : {original_count:>8,}")
print(f"  Cleaned rows  : {len(df_clean):>8,}")
print(f"  Rows removed  : {original_count - len(df_clean):>8,}")
print("\n  Removal breakdown:")
for reason, n in log:
    flag = "!" if n > 0 else " "
    print(f"  {flag} [{n:>5,}]  {reason}")
print(f"\n  Output: {OUTPUT_FILE}")
print("=" * 65)

tw     = df_clean[df_clean["Is_Taiwan"]]
non_tw = df_clean[~df_clean["Is_Taiwan"]]

gross_revenue = df_clean["Revenue_USD"].sum()
net_revenue   = df_clean[df_clean["Status"].isin({"Shipped", "Pending"})]["Revenue_USD"].sum()
tw_gross      = tw["Revenue_USD"].sum()
tw_net        = tw[tw["Status"].isin({"Shipped", "Pending"})]["Revenue_USD"].sum()

print("\n")
print("=" * 65)
print("  iGS GlobalRetail — REVENUE REPORT 2026")
print("=" * 65)
print(f"  FX Rate      : 1 NTD = 1/32 USD (= ${FX_NTD_TO_USD:.6f})")
print(f"  Gross Revenue: all statuses (Shipped + Pending + Returned + Cancelled)")
print(f"  Net Revenue  : Shipped + Pending only")
print(f"  Taiwan region: orders originally denominated in NTD")
print("=" * 65)

print(f"\n  Revenue by Status  (ALL REGIONS)")
print(f"  {'─' * 55}")
for status in ["Shipped", "Pending", "Returned", "Cancelled"]:
    sub = df_clean[df_clean["Status"] == status]
    rev = sub["Revenue_USD"].sum()
    pct = rev / gross_revenue * 100 if gross_revenue else 0
    print(f"  {status:10} : ${rev:>12,.0f}  ({pct:5.1f}%)  [{len(sub):,} orders]")
print(f"  {'─' * 55}")
print(f"  {'TOTAL':10} : ${gross_revenue:>12,.0f}  (100.0%)  [{len(df_clean):,} orders]")

print(f"\n  Revenue by Product Category  (ALL REGIONS)")
print(f"  {'─' * 55}")
cat_rev = df_clean.groupby("Product_Category")["Revenue_USD"].sum().sort_values(ascending=False)
for cat, rev in cat_rev.items():
    pct = rev / gross_revenue * 100 if gross_revenue else 0
    print(f"  {cat:15} : ${rev:>12,.0f}  ({pct:5.1f}%)")
print(f"  {'─' * 55}")
print(f"  {'TOTAL':15} : ${gross_revenue:>12,.0f}  (100.0%)")

print(f"\n  Revenue by Status  (TAIWAN ONLY — NTD orders)")
print(f"  {'─' * 55}")
for status in ["Shipped", "Pending", "Returned", "Cancelled"]:
    sub = tw[tw["Status"] == status]
    rev = sub["Revenue_USD"].sum()
    pct = rev / tw_gross * 100 if tw_gross else 0
    print(f"  {status:10} : ${rev:>12,.0f}  ({pct:5.1f}%)  [{len(sub):,} orders]")
print(f"  {'─' * 55}")
print(f"  {'TOTAL':10} : ${tw_gross:>12,.0f}  (100.0%)  [{len(tw):,} orders]")

print(f"\n  Revenue by Product Category  (TAIWAN ONLY)")
print(f"  {'─' * 55}")
tw_cat_rev = tw.groupby("Product_Category")["Revenue_USD"].sum().sort_values(ascending=False)
for cat, rev in tw_cat_rev.items():
    pct = rev / tw_gross * 100 if tw_gross else 0
    print(f"  {cat:15} : ${rev:>12,.0f}  ({pct:5.1f}%)")
print(f"  {'─' * 55}")
print(f"  {'TOTAL':15} : ${tw_gross:>12,.0f}  (100.0%)")

unit_mean   = df_clean['Unit_Sold'].mean()
unit_median = df_clean['Unit_Sold'].median()
unit_mode   = df_clean['Unit_Sold'].mode()
unit_total  = df_clean['Unit_Sold'].sum()

print(f"\n  Units per Order  (ALL REGIONS)")
print(f"  {'─' * 55}")
print(f"  Mean   : {unit_mean:.2f}"     if not pd.isna(unit_mean)   else "  Mean   : N/A")
print(f"  Median : {unit_median:.0f}"   if not pd.isna(unit_median) else "  Median : N/A")
print(f"  Mode   : {unit_mode.iloc[0]}" if not unit_mode.empty      else "  Mode   : N/A")
print(f"  Total  : {unit_total:,}")

print("=" * 65)
print(f"  GROSS REVENUE — All Regions  : ${gross_revenue:>12,.0f}")
print(f"  NET REVENUE   — All Regions  : ${net_revenue:>12,.0f}")
print(f"  {'─' * 55}")
print(f"  GROSS REVENUE — Taiwan Only  : ${tw_gross:>12,.0f}  ({tw_gross/gross_revenue*100:.1f}% of total)" if gross_revenue else "  GROSS REVENUE — Taiwan Only  : N/A")
print(f"  NET REVENUE   — Taiwan Only  : ${tw_net:>12,.0f}  ({tw_net/net_revenue*100:.1f}% of net)"         if net_revenue   else "  NET REVENUE   — Taiwan Only  : N/A")
print("=" * 65)

total_orders    = len(df_clean)
tw_orders       = len(tw)
non_tw_orders   = len(non_tw)

returned_all    = (df_clean["Status"] == "Returned").sum()
returned_tw     = (tw["Status"] == "Returned").sum()
returned_non_tw = (non_tw["Status"] == "Returned").sum()

return_rate_all    = returned_all    / total_orders    * 100 if total_orders    else 0
return_rate_tw     = returned_tw     / tw_orders       * 100 if tw_orders       else 0
return_rate_non_tw = returned_non_tw / non_tw_orders   * 100 if non_tw_orders   else 0

avg_units_all    = df_clean["Unit_Sold"].mean()
avg_units_tw     = tw["Unit_Sold"].mean()     if tw_orders     else 0.0
avg_units_non_tw = non_tw["Unit_Sold"].mean() if non_tw_orders else 0.0

print("\n")
print("=" * 65)
print("  Order Count")
print("=" * 65)
print(f"  {'─' * 55}")
print(f"  {'Region':<20}  {'Orders':>8}  {'Share':>8}")
print(f"  {'─' * 55}")
print(f"  {'All Regions':<20}  {total_orders:>8,}  {'100.0%':>8}")
print(f"  {'  Taiwan (NTD)':<20}  {tw_orders:>8,}  {tw_orders/total_orders*100:>7.1f}%" if total_orders else "  N/A")
print(f"  {'  Non-Taiwan':<20}  {non_tw_orders:>8,}  {non_tw_orders/total_orders*100:>7.1f}%" if total_orders else "  N/A")
print(f"  {'─' * 55}")

print(f"\n  Order Count by Status  (ALL REGIONS)")
print(f"  {'─' * 55}")
print(f"  {'Status':<12}  {'Orders':>8}  {'Share':>8}")
print(f"  {'─' * 55}")
for status in ["Shipped", "Pending", "Returned", "Cancelled"]:
    n   = (df_clean["Status"] == status).sum()
    pct = n / total_orders * 100 if total_orders else 0
    print(f"  {status:<12}  {n:>8,}  {pct:>7.1f}%")
print(f"  {'─' * 55}")
print(f"  {'TOTAL':<12}  {total_orders:>8,}  {'100.0%':>8}")

print(f"\n  Order Count by Product Category  (ALL REGIONS)")
print(f"  {'─' * 55}")
print(f"  {'Category':<16}  {'Orders':>8}  {'Share':>8}")
print(f"  {'─' * 55}")
cat_counts = df_clean["Product_Category"].value_counts().sort_index()
for cat, n in cat_counts.items():
    pct = n / total_orders * 100 if total_orders else 0
    print(f"  {cat:<16}  {n:>8,}  {pct:>7.1f}%")
print(f"  {'─' * 55}")
print(f"  {'TOTAL':<16}  {total_orders:>8,}  {'100.0%':>8}")

print("\n\n")
print("=" * 65)
print("  Return Rate")
print("  Definition: Return Rate = Returned orders / Total orders")
print("=" * 65)
print(f"  {'─' * 55}")
print(f"  {'Region':<20}  {'Returned':>8}  {'Total':>8}  {'Rate':>8}")
print(f"  {'─' * 55}")
print(f"  {'All Regions':<20}  {returned_all:>8,}  {total_orders:>8,}  {return_rate_all:>7.1f}%")
print(f"  {'  Taiwan (NTD)':<20}  {returned_tw:>8,}  {tw_orders:>8,}  {return_rate_tw:>7.1f}%")
print(f"  {'  Non-Taiwan':<20}  {returned_non_tw:>8,}  {non_tw_orders:>8,}  {return_rate_non_tw:>7.1f}%")
print(f"  {'─' * 55}")

print(f"\n  Return Rate by Product Category  (ALL REGIONS)")
print(f"  {'─' * 55}")
print(f"  {'Category':<16}  {'Returned':>8}  {'Total':>8}  {'Rate':>8}")
print(f"  {'─' * 55}")
for cat in sorted(df_clean["Product_Category"].unique()):
    sub     = df_clean[df_clean["Product_Category"] == cat]
    n_ret   = (sub["Status"] == "Returned").sum()
    n_total = len(sub)
    rate    = n_ret / n_total * 100 if n_total else 0
    print(f"  {cat:<16}  {n_ret:>8,}  {n_total:>8,}  {rate:>7.1f}%")
print(f"  {'─' * 55}")
print(f"  {'TOTAL':<16}  {returned_all:>8,}  {total_orders:>8,}  {return_rate_all:>7.1f}%")

print("\n\n")
print("=" * 65)
print("  Avg Units per Order")
print("=" * 65)
print(f"  {'─' * 55}")
print(f"  {'Region':<20}  {'Avg Units/Order':>16}")
print(f"  {'─' * 55}")
print(f"  {'All Regions':<20}  {avg_units_all:>16.2f}")
print(f"  {'  Taiwan (NTD)':<20}  {avg_units_tw:>16.2f}")
print(f"  {'  Non-Taiwan':<20}  {avg_units_non_tw:>16.2f}")
print(f"  {'─' * 55}")

print(f"\n  Avg Units per Order by Status  (ALL REGIONS)")
print(f"  {'─' * 55}")
print(f"  {'Status':<12}  {'Avg Units/Order':>16}")
print(f"  {'─' * 55}")
for status in ["Shipped", "Pending", "Returned", "Cancelled"]:
    sub = df_clean[df_clean["Status"] == status]
    avg = sub["Unit_Sold"].mean() if len(sub) > 0 else 0.0
    print(f"  {status:<12}  {avg:>16.2f}")
print(f"  {'─' * 55}")
print(f"  {'OVERALL':<12}  {avg_units_all:>16.2f}")

print(f"\n  Avg Units per Order by Product Category  (ALL REGIONS)")
print(f"  {'─' * 55}")
print(f"  {'Category':<16}  {'Avg Units/Order':>16}  {'Total Units':>12}")
print(f"  {'─' * 55}")
cat_avg = (df_clean.groupby("Product_Category")["Unit_Sold"]
           .agg(["mean", "sum"])
           .sort_values("mean", ascending=False))
for cat, row in cat_avg.iterrows():
    print(f"  {cat:<16}  {row['mean']:>16.2f}  {int(row['sum']):>12,}")
print(f"  {'─' * 55}")
print(f"  {'OVERALL':<16}  {avg_units_all:>16.2f}  {df_clean['Unit_Sold'].sum():>12,}")
print("=" * 65)
