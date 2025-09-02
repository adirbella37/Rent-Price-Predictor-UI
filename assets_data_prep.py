import numpy as np
import pandas as pd
import re
import requests
import joblib


# ---------------------------------------------------------------------
# Data Preprocessing Functions
# ---------------------------------------------------------------------
def drop_duplicates_except_description_and_images(df):
    
    '''
    This function is designed to identify and remove duplicate listings based on all features except for the 'description'
    and 'num_of_images' columns.
    These two fields are excluded from the comparison because duplicate apartments often have slight variations in wording
    or image count, even when all other data is identical.
    By ignoring them, we preserve only the first occurrence of each truly unique listing.
    '''
    
    cols = ['description', 'num_of_images']
    tmp = df.drop(columns=[c for c in cols if c in df.columns], errors='ignore')
    
    return df.loc[~tmp.duplicated(keep='first')]


def drop_missing_neighborhood(df):

    '''
    Removes rows with missing values in the 'neighborhood' column, as these are typically advertisements rather than actual listings.
    Excluding them helps maintain the reliability and integrity of the dataset.
    '''
    return df.dropna(subset=['neighborhood'])


def update_floor_and_total(df):

    '''
    This function cleans and standardizes the 'floor' column, which contains values scraped from the web
    and may appear in inconsistent textual formats.

    It handles the following cases:
    'קרקע מתוך קרקע' -> '0 בקומה ו- 0 בסך קומות'
    'קרקע מתוך מספר' -> '0 בקומה ומספר בסך קומות'
    'מספר מתוך מספר' -> 'מספר שכתוב מימין למילה 'מתוך' לעמודת קומה ומספר שכתוב משמאל למילה 'מתוך' לעמודת סך קומות'

    This transformation ensures that both the current floor and total_floors are extracted
    and stored as numeric values in separate columns.
    '''

    df = df.copy()
    
    for i, v in enumerate(df['floor']):
        if pd.isna(v):
            continue
            
        s = str(v).strip()
        
        if s == "קרקע מתוך קרקע":
            df.loc[i, ['floor', 'total_floors']] = [0, 0]
            
        elif re.match(r'^קרקע\s?מתוך\s?\d+$', s):
            y = int(re.findall(r'\d+', s)[0])
            df.loc[i, ['floor', 'total_floors']] = [0, y]
            
        elif re.match(r'^\d+\s?מתוך\s?\d+$', s):
            x, y = map(int, re.findall(r'\d+', s))
            df.loc[i, ['floor', 'total_floors']] = [x, y]
            
    return df


def fix_missing_total_floors(df):
    
    '''
    This function corrects missing values in the 'total_floors' column based on patterns we identified in the dataset:
    If 'floor' is 0 and 'total_floors' is missing, we assume it's a one-level building and set 'total_floors' to 0.
    If 'floor' is greater than 0 and 'total_floors' is missing, we found that in many cases the values were mistakenly swapped.
    Therefore, we move the value from 'floor' to 'total_floors' and set 'floor' to 0.
    '''
    
    df = df.copy()
    
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    df['total_floors'] = pd.to_numeric(df['total_floors'], errors='coerce')

    mask0 = (df['floor'] == 0) & df['total_floors'].isna()
    df.loc[mask0, 'total_floors'] = 0

    mask_swap = (df['floor'].notna()) & (df['floor'] > 0) & df['total_floors'].isna()
    df.loc[mask_swap, ['total_floors', 'floor']] = df.loc[mask_swap, ['floor', 'total_floors']].fillna(0).values


    df[['floor', 'total_floors']] = df[['floor', 'total_floors']].astype('Int64')
    
    return df


def floor_from_conditions(df):
    
    '''
    This function fills in missing values for 'floor' and 'total_floors' based on textual patterns in the property description
    and specific known addresses.

    We identified that in many cases where 'total_floors' is missing, the floor number can be extracted
    from the description using a pattern like "קומה X".

    Additionally, for certain addresses with completely missing values for both 'floor' and 'total_floors',
    we manually inferred the correct values based on external sources.

    These corrections improve the quality and completeness of the floor-related fields in the dataset.
    '''

    df = df.copy()
    
    for idx, row in df.loc[df['total_floors'].isna()].iterrows():
        desc = str(row.get('description', '')).lower()
        address = row.get('address', '')
        price = row.get('price', np.nan)

        m = re.search(r'קומה\s*(\d+)', desc)
        if m:
            df.loc[idx, 'floor'] = int(m.group(1))

        if address == "ליאונרדו דה וינצ'י 14":
            df.loc[idx, ['floor', 'total_floors']] = [15, 40] if pd.notna(price) and price < 15000 else [22, 40]
            
        elif address == "ניסים אלוני 21":
            df.loc[idx, ['floor', 'total_floors']] = [22, 35]
            
    return df


def floor_from_conditions_for_test(df):
    
    '''
    This function is the same as the one above, with one key difference:
    it removes the use of the 'price' column in order to prevent data leakage during test set processing.
    '''

    df = df.copy()
    
    for idx, row in df.loc[df['total_floors'].isna()].iterrows():
        desc = str(row.get('description', '')).lower()
        address = str(row.get('address', '')).strip()

        m = re.search(r'קומה\s*(\d+)', desc)
        if m:
            df.loc[idx, 'floor'] = int(m.group(1))

        if address == "ליאונרדו דה וינצ'י 14":
            df.loc[idx, ['floor', 'total_floors']] = [22, 40]
            
        elif address == "ניסים אלוני 21":
            df.loc[idx, ['floor', 'total_floors']] = [22, 35]
            
    return df


def fix_floor_ending_with_total(df):
    
    '''
    This function fixes invalid 'floor' values that result from incorrect data extraction.

    We identified a pattern where the 'floor' value incorrectly concatenates the actual floor number with the total number of floors.
    For example, a value like 123 should be interpreted as floor 3 & toal_floors 12.

    The function detects such cases where 'floor' ends with the value in 'total_floors' and is numerically greater, 
    and extracts the correct floor number by removing the suffix that matches the total.
    '''

    def correct(row):
        f, t = row['floor'], row['total_floors']
        
        if pd.notna(f) and pd.notna(t):
            fs, ts = str(f), str(t)
            
            if f > t and fs.endswith(ts):
                try:
                    return int(fs[:-len(ts)])
                
                except ValueError:
                    return f
                
        return f

    df = df.copy()
    
    df['floor'] = df.apply(correct, axis=1)
    
    return df


def fix_invalid_floor_values(df):
    
    '''
    This function handles a broader range of invalid cases in the 'floor' and 'total_floors' columns that result from
    incorrect data extraction.

    It covers situations where:
    - 'floor' > 0 and 'total_floors' is 0 -> the values are likely swapped, so they are reversed.
    - 'floor' contains both the actual floor and total floors concatenated (e.g., 31 or 123) -> we split the number into 
    its components: the smaller part becomes 'floor' and the larger part becomes 'total_floors'.

    Unlike the previous function, which targets a specific pattern where the floor ends with the total,
    this function provides more general coverage for structurally invalid floor data.
    '''

    df = df.copy()
    
    valid = df['floor'].notna() & df['total_floors'].notna()
    cond = df.loc[valid, 'floor'].astype(float) > df.loc[valid, 'total_floors'].astype(float)

    for i, row in df.loc[valid & cond].iterrows():
        f, t = row['floor'], row['total_floors']
        
        if t == 0 and f > 0:
            df.loc[i, ['floor', 'total_floors']] = [0, f]
            
        elif t == 1 and f >= 10:
            fs = str(int(f))
            
            if len(fs) == 2:
                df.loc[i, ['floor', 'total_floors']] = [1, int(fs[1])]
                
            elif len(fs) == 3:
                df.loc[i, ['floor', 'total_floors']] = [int(fs[:-1]), int(fs[-1])]
                
    return df


def clean_property_type(df):
    
    '''
    This function standardizes the values in the 'property_type' column by unifying similar categories under a single label.

    It removes extra characters and whitespace, and maps variations of the same property type (e.g., "דירה להשכרה", "Квартира")
    into a consistent form (e.g., "דירה").

    Additionally, it removes rows labeled as "סאבלט", which were considered non-representative and irrelevant for analysis.
    '''

    df = df.copy()
    
    df['property_type'] = df['property_type'].astype(str).str.strip().str.replace('\u200e', '', regex=True)
    
    replacer = {
        'גג/ פנטהאוז': 'גג/פנטהאוז',
        'גג/פנטהאוז להשכרה': 'גג/פנטהאוז',
        'דירת גן להשכרה': 'דירת גן',
        'דירה להשכרה': 'דירה',
        'באתר מופיע ערך שלא ברשימה הסגורה': 'דירה',
        'החלפת דירות': 'דירה',
        'מרתף/פרטר': 'דירה',
        'Квартира': 'דירה',
        "בית פרטי/ קוטג'": "פרטי/קוטג'",
        'דו משפחתי': "פרטי/קוטג'"
    }
    
    df['property_type'] = df['property_type'].replace(replacer)
    
    return df[df['property_type'] != 'סאבלט']


def remove_general_with_keywords(df):
    
    '''
    This function removes listings that are not relevant to residential apartment rentals.

    Specifically, it filters out listings where the property type is "כללי" and the description contains keywords such as
    "חנות" or "דרושים".
    '''

    keywords = ['חנות', 'דרושים']
    
    mask = (df['property_type'] == 'כללי') & df['description'].fillna('').str.contains('|'.join(keywords))
    
    return df[~mask]


def update_room_num_from_description_conditional(row):
    
    '''
    This function corrects missing or invalid room numbers using information extracted from the description.

    If the 'room_num' field is valid (not missing and not zero), it is returned as-is.

    Otherwise, the function uses regular expressions to identify room count patterns within the listing's description,
    such as numeric values followed by the word "חדרים", or phrases like "חד'", "דירה חדר", or "חדר וחצי".

    It returns the best-estimated room count based on these textual clues.
    '''

    if not (pd.isna(row['room_num']) or row['room_num'] == 0):
        return row['room_num']

    desc = str(row['description']).lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*חדרים?', desc)
    
    if m:
        return float(m.group(1)) if '.' in m.group(1) else int(m.group(1))

    if re.search(r"\bחד'|\bדירה חדר\b", desc):
        return 1
    
    if re.search(r'(חדר וחצי|1\.5 חדרים)', desc):
        return 1.5
    
    return row['room_num']


def fill_missing_numeric_by_median(df):
    
    '''
    This function fills missing or zero values in selected numeric columns using the median.

    It is specifically designed for columns like 'monthly_arnona' and 'building_tax', where missing or zero values are replaced with the median value within each neighborhood group.

    If no valid (non-zero) values exist in a specific group, the function falls back to the overall median from the entire dataset.
    '''

    df = df.copy()
    
    for col in ['monthly_arnona', 'building_tax']:
        def repl(s):
            med = s[s != 0].median()
            
            if pd.isna(med):
                med = df.loc[(df[col] != 0) & df[col].notna(), col].median()
                
            return s.apply(lambda v: med if (pd.isna(v) or v == 0) else v)

        df[col] = df.groupby('neighborhood')[col].transform(repl)
        
    return df


def clean_and_filter_area(df):
    
    '''
    This function filters out entries with an 'area' value below 10.

    After reviewing the dataset, we identified that values below 10 are not realistic for apartment listings.
    Such rows typically represent non-residential units like parking spots or storage rooms that were not fully filtered
    out by earlier steps.
    '''

    return df[df['area'].notna() & (df['area'] >= 10)]


def fix_area_from_description(df):
    
    '''
    This function corrects unusually large 'area' values for apartment listings by extracting more accurate size information
    from the description.

    If the property type is classified as an apartment ("דירה") and its recorded area exceeds 400 square meters,
    we assume the value is likely incorrect, since typical apartments are rarely that large.

    In such cases, we attempt to extract a more reasonable area value using a regular expression pattern that matches numbers
    followed by square meter units within the description.
    '''

    df = df.copy()
    
    pat = r'(\d{2,4})\s*(?:מר|מ"ר|מטר|מטרים)'
    
    for idx, row in df.iterrows():
        area, ptype, desc = row.get('area'), row.get('property_type', ''), str(row.get('description', ''))
        
        if pd.notna(area) and area > 400 and str(ptype).strip() == 'דירה':
            m = re.search(pat, desc)
            
            if m:
                df.at[idx, 'area'] = int(m.group(1))
                
    return df



def normalize_distance_column(df):
    
    '''
    This function normalizes the 'distance_from_center' column by converting values from meters to kilometers.

    Only values greater than 50 meters are converted. The final result is rounded to 4 decimal places for consistency.
    '''

    df = df.copy()
    
    df['distance_from_center'] = df['distance_from_center'].apply(
        lambda x: x / 1000 if pd.notna(x) and x > 50 else x)
    
    df['distance_from_center'] = df['distance_from_center'].round(4)
    
    return df


def classify_general_properties_by_order(df):
    
    '''
    This function refines listings where the property type is labeled as "כללי" by reclassifying or removing entries based
    on patterns found in the description and other features.

    The logic includes:
    - If the description contains "חניה" and both floor and total_floors are missing -> drop the row.
    - If the description contains "סטודיו" -> reclassify as "סטודיו/לופט".
    - If the description contains "דירה"/"דירת" and total_floors > 0 -> reclassify as "דירה".
    - If area ≤ 65, room number ≤ 2, and both floor and total_floors are 0 -> reclassify as "יחידת דיור".
    - If the description contains "מתפנה", "חדר", or "משרד" -> drop the row.

    This rule-based cleanup improves property type classification for listings that were initially labeled too broadly.
    '''

    df = df.copy()
    
    drop_rows = []
    
    for idx, row in df.iterrows():
        if str(row.get('property_type', '')).strip() != 'כללי':
            continue

        desc = str(row.get('description', '')).lower()
        floor, total = row.get('floor'), row.get('total_floors')
        area, rooms = row.get('area'), row.get('room_num')

        if 'חניה' in desc and pd.isna(floor) and pd.isna(total):
            drop_rows.append(idx)
            continue

        if 'סטודיו' in desc:
            df.at[idx, 'property_type'] = 'סטודיו/לופט'
            continue

        if desc and pd.notna(total) and total > 0 and any(w in desc for w in ['דירה', 'דירת']):
            df.at[idx, 'property_type'] = 'דירה'
            continue

        if pd.notna(area) and area <= 65 and pd.notna(rooms) and rooms <= 2 and floor == 0 and total == 0:
            df.at[idx, 'property_type'] = 'יחידת דיור'
            continue

        if any(w in desc for w in ['מתפנה', 'חדר']) or 'משרד' in desc:
            drop_rows.append(idx)
            
    return df.drop(index=drop_rows)


def replace_distance_above_threshold_with_neighborhood_median(df):
    
    '''
    This function handles outliers in the 'distance_from_center' column after converting distances to kilometers.

    Based on data review, listings with a distance greater than 30 km were considered unrealistic.

    For such cases, the function replaces the distance with the median distance of other listings in the same neighborhood.
    '''

    df = df.copy()
    
    for idx, row in df.loc[df['distance_from_center'] > 30].iterrows():
        neigh = row['neighborhood']
        med = df.loc[(df['neighborhood'] == neigh) &
                     (df['distance_from_center'] <= 30) &
                     df['distance_from_center'].notna(),
                     'distance_from_center'].median()
        
        if pd.notna(med):
            df.at[idx, 'distance_from_center'] = med
            
    return df


def clean_and_filter_properties(df):
    
    '''
    This function filters out real estate listings with unrealistic or inconsistent values, based on combinations of price, property type, area, and description.

    It removes:
    - Listings with extremely high prices (e.g., over 100,000).
    - Listings where the price-to-size ratio is suspicious (e.g., a large cottage with a very low price, or a tiny studio with a very high price).
    - Luxury apartments or penthouses with a high price but no luxury-related keywords in the description.

    For regular apartments labeled as "דירה", if the price is missing or zero,
    the function attempts to extract a price from the description using a regular expression.

    This step helps improve the quality of the dataset by eliminating outliers and correcting missing pricing when possible.
    '''

    df = df.copy()
    
    drops = []

    def ext_price(desc):
        m = re.search(r'(\d{1,3}(?:[,.\s]?\d{3})*)\s*(שקלים|ש"ח|שקל|₪)', desc)
        
        if m:
            return float(m.group(1).replace(',', '').replace(' ', '').replace('.', ''))
        return None

    for idx, row in df.iterrows():
        ptype = str(row.get('property_type', '')).strip()
        area, price = row.get('area'), row.get('price')
        floor, total = row.get('floor'), row.get('total_floors')
        desc = str(row.get('description', '')).lower()

        if pd.notna(price) and price > 100_000:
            drops.append(idx)
            continue

        if ptype == "פרטי/קוטג'" and pd.notna(area) and area > 200 and pd.notna(price) and price < 20000:
            drops.append(idx)
            continue

        if (ptype.lower() == "דופלקס" and pd.notna(floor) and pd.notna(total) and
                floor > 0 and total > 0 and floor == total and pd.notna(price) and price < 11000):
            drops.append(idx)
            continue

        if ptype == "יחידת דיור" and pd.notna(area) and area <= 30 and pd.notna(price) and price > 4200:
            drops.append(idx)
            continue

        if ptype == "דירת גן" and pd.notna(area) and area > 50 and pd.notna(price) and price < 6000:
            drops.append(idx)
            continue

        if ptype == "דירה":
            new_p = ext_price(desc)
            
            if new_p is not None and (pd.isna(price) or price == 0):
                df.at[idx, 'price'] = new_p
            else:
                lux = ['יוקרתית', 'היוקרתי', 'מפוארת', 'והמפואר']
                
                if pd.notna(price) and price > 30000 and not any(w in desc for w in lux):
                    drops.append(idx)

        if ptype == "גג/פנטהאוז":
            lux = ['יוקרתית', 'היוקרתי', 'מפוארת', 'והמפואר']
            
            if pd.notna(price) and price > 30000 and not any(w in desc for w in lux):
                drops.append(idx)

    return df.drop(index=drops)


def drop_large_area_low_price(df):

    '''
    This function removes outlier listings with a large area and an unusually low price.

    After analyzing the dataset, we concluded that it's highly unlikely for a valid apartment 
    to have an area greater than 100 square meters and a price below 5000.

    Based on this insight, we decided to remove such entries to improve data quality.
    '''


    return df[~((df['area'] >= 100) & (df['price'] <= 5000))]


def drop_apartments_price_below_arnona(df):
    
    '''
    This function removes apartment listings where the monthly rent is lower than the monthly municipal tax (arnona).

    Such cases were identified as likely errors or non-residential listings, as it's highly unlikely for the rent to be
    lower than the property tax.
    '''

    mask = (df['property_type'] == 'דירה') & (df['price'] < df['monthly_arnona'])
    
    return df[~mask]


def replace_building_tax_above_threshold_with_neighborhood_median(df):
    
    '''
    This function identifies listings where the building maintenance fee (building_tax) exceeds a reasonable threshold.

    Based on data analysis and external references, we concluded that monthly building fees above 3000₪ are unrealistic for residential apartments.

    For each such case, the value is replaced with the median building tax of the same neighborhood, calculated only from listings with valid and reasonable values.

    This helps reduce the impact of outliers and ensures more realistic building tax values across the dataset.
    '''


    df = df.copy()
    
    for idx, row in df.loc[df['building_tax'] > 3000].iterrows():
        neigh = row['neighborhood']
        med = df.loc[(df['neighborhood'] == neigh) &
                     (df['building_tax'] <= 3000) &
                     df['building_tax'].notna(),
                     'building_tax'].median()
        
        if pd.notna(med):
            df.at[idx, 'building_tax'] = med
            
    return df


def drop_rows_price_below_1500(df):
    
    '''
    This function removes listings with a price below 1500₪.

    Based on data review, such prices are considered unrealistic for residential properties and are likely to reflect data entry errors or irrelevant listings.
    '''

    return df[df['price'] >= 1500]



def fill_missing_distance(df, distance_info):
    df = df.copy()
    medians = distance_info["neighborhood_medians"]
    fallback = distance_info["fallback"]

    mask = df['distance_from_center'].isna() | (df['distance_from_center'] == 0)

    for idx, row in df.loc[mask].iterrows():
        neigh = row.get('neighborhood')

        # שכונה חסרה או median שלה לא מוגדר → fallback
        if pd.isna(neigh) or pd.isna(medians.get(neigh)):
            dist = fallback
        else:
            dist = medians[neigh]

        df.at[idx, 'distance_from_center'] = dist

    return df





# ---------------------------------------------------------------------
# prepare_data – MAIN
# ---------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    if dataset_type not in {"train", "test"}:
        raise ValueError("dataset_type must be 'train' or 'test'")

    # ---------- Basic Cleaning ----------
    df = df.drop_duplicates()
    df = drop_duplicates_except_description_and_images(df)
    df = drop_missing_neighborhood(df)
    df = update_floor_and_total(df)
    df = fix_missing_total_floors(df)
    df = floor_from_conditions(df) if dataset_type == "train" else floor_from_conditions_for_test(df)
    df = fix_floor_ending_with_total(df)
    df = fix_invalid_floor_values(df)
    df = clean_property_type(df)
    df = remove_general_with_keywords(df)
    df = df.drop(columns=['num_of_images', 'num_of_payments', 'days_to_enter'], errors='ignore')

    df['room_num'] = df.apply(update_room_num_from_description_conditional, axis=1)
    df = df[~((df['room_num'] == 0) | df['room_num'].isna())].reset_index(drop=True)

    df = fill_missing_numeric_by_median(df)
    df = clean_and_filter_area(df)
    df = fix_area_from_description(df)

    df['garden_area'] = df['garden_area'].fillna(0)
    df = df[~df['property_type'].isin(['מחסן', 'חניה'])]
    df = normalize_distance_column(df)
    df = classify_general_properties_by_order(df)

    # ---------- Distance Cleanup ----------
    df = replace_distance_above_threshold_with_neighborhood_median(df)

    # חישוב מדיאנים (רק לאחר תיקון חריגים) ושמירה
    if dataset_type == "train":
        distance_info = {
            "neighborhood_medians": df.groupby('neighborhood')['distance_from_center'].median().to_dict(),
            "fallback": df['distance_from_center'].median()
        }
        joblib.dump(distance_info, "neigh_dist_medians.pkl")
    else:
        distance_info = joblib.load("neigh_dist_medians.pkl")

    df = fill_missing_distance(df, distance_info)
    df = replace_building_tax_above_threshold_with_neighborhood_median(df)

    # ---------- Train Only Filters ----------
    if dataset_type == "train":
        df = df[df['price'].notna()]
        df = clean_and_filter_properties(df)
        df = drop_large_area_low_price(df)
        df = drop_apartments_price_below_arnona(df)
        df = drop_rows_price_below_1500(df)

    # ---------- Feature Engineering ----------
    df['arnona_per_m2'] = df['monthly_arnona'] / df['area'].replace(0, np.nan)
    df['area_x_elevator'] = df['area'] * df['elevator']
    df['area_per_room'] = df['area'] / df['room_num'].replace(0, np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['arnona_per_m2', 'area_x_elevator', 'area_per_room'], inplace=True)

    # ---------- IQR Outliers ----------
    if dataset_type == "train":
        q1, q3 = df['price'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df['price'] >= q1 - 1.5 * iqr) & (df['price'] <= q3 + 1.5 * iqr)]
        price_col = df.pop('price')
        df['price'] = price_col

    return df.reset_index(drop=True)



