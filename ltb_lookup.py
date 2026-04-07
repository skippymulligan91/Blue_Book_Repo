# ltb_lookup.py
# LTB buckling resistance lookup for UB sections per SCI Blue Book
# Supports S275 (S355 ready via url parameter)
# David Mulligan — Blyth & Blyth

import pandas as pd
import numpy as np
from io import StringIO
from scipy.interpolate import interp1d

# ── Raw GitHub URLs ──────────────────────────────────────────────────────────
CSV_URL_S275 = (
    "https://raw.githubusercontent.com/skippymulligan91/"
    "Blue_Book_Repo/refs/heads/main/UB_LTB_Capacity_S275.csv"
)

# Add S355 URL here when CSV is uploaded to repo:
# CSV_URL_S355 = (
#     "https://raw.githubusercontent.com/skippymulligan91/"
#     "Blue_Book_Repo/refs/heads/main/UB_LTB_Capacity_S355.csv"
# )


# ── Module 1a — Load from URL (local Jupyter) ────────────────────────────────
def load_ltb_table(url: str = CSV_URL_S275) -> pd.DataFrame:
    """
    Load LTB capacity table directly from GitHub raw CSV URL.
    Use this in your local Jupyter notebook.

    Parameters
    ----------
    url : str
        Raw GitHub URL to the CSV file (default S275 table)

    Returns
    -------
    pd.DataFrame
        Full table with columns: Section, C1, L=1.0 ... L=14.0, Iy_cm4
    """
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df


# ── Module 1b — Load from string (Claude sandbox) ───────────────────────────
def load_ltb_table_from_string(csv_text: str) -> pd.DataFrame:
    """
    Load LTB capacity table from a raw CSV string.
    Used when pandas cannot reach GitHub directly (e.g. Claude sandbox).
    Claude fetches the CSV via web_fetch and passes the text here.

    Parameters
    ----------
    csv_text : str
        Raw CSV content as a string

    Returns
    -------
    pd.DataFrame
        Full table with columns: Section, C1, L=1.0 ... L=14.0, Iy_cm4
    """
    df = pd.read_csv(StringIO(csv_text))
    df.columns = df.columns.str.strip()
    return df


# ── Module 2 — Filter by section and C1 ─────────────────────────────────────
def get_ltb_row(df: pd.DataFrame, section: str, C1: float) -> pd.Series:
    """
    Extract LTB capacity row for a given section and C1 value.
    If C1 falls between two tabulated values, linearly interpolates
    between the two bounding rows.

    Parameters
    ----------
    df : pd.DataFrame
        Full LTB table from load_ltb_table() or load_ltb_table_from_string()
    section : str
        Section designation e.g. "457 x 191 x 82"
    C1 : float
        Moment gradient factor e.g. 1.0, 1.35 etc.

    Returns
    -------
    pd.Series
        Row of Mb,Rd capacities (kNm) indexed by length e.g. "L=1.0" ... "L=14.0"
    """
    df_sec = df[df["Section"] == section].copy()

    if df_sec.empty:
        raise ValueError(
            f"Section '{section}' not found in table. "
            f"Check designation format e.g. '457 x 191 x 82'"
        )

    c1_vals  = sorted(df_sec["C1"].unique())
    len_cols = [c for c in df.columns if c.startswith("L=")]

    # Exact C1 match
    if C1 in c1_vals:
        return df_sec[df_sec["C1"] == C1][len_cols].iloc[0]

    # C1 below minimum — clamp conservatively
    if C1 < c1_vals[0]:
        print(f"  Warning: C1={C1} below minimum tabulated {c1_vals[0]}. "
              f"Using C1={c1_vals[0]} (conservative)")
        return df_sec[df_sec["C1"] == c1_vals[0]][len_cols].iloc[0]

    # C1 above maximum — clamp at top
    if C1 > c1_vals[-1]:
        print(f"  Warning: C1={C1} above maximum tabulated {c1_vals[-1]}. "
              f"Using C1={c1_vals[-1]}")
        return df_sec[df_sec["C1"] == c1_vals[-1]][len_cols].iloc[0]

    # Interpolate between two bounding C1 rows
    c1_lo  = max(v for v in c1_vals if v < C1)
    c1_hi  = min(v for v in c1_vals if v > C1)
    row_lo = df_sec[df_sec["C1"] == c1_lo][len_cols].iloc[0]
    row_hi = df_sec[df_sec["C1"] == c1_hi][len_cols].iloc[0]
    t      = (C1 - c1_lo) / (c1_hi - c1_lo)

    return row_lo + t * (row_hi - row_lo)


# ── Module 3 — Interpolate for given length ──────────────────────────────────
def interpolate_ltb_capacity(row: pd.Series, L: float) -> float:
    """
    Interpolate Mb,Rd for a given unrestrained length from a capacity row.
    Uses scipy linear interpolation between tabulated lengths.

    Parameters
    ----------
    row : pd.Series
        Capacity row from get_ltb_row() indexed by "L=1.0", "L=1.5" etc.
    L : float
        Unrestrained length (m) to get capacity at

    Returns
    -------
    float
        Interpolated Mb,Rd (kNm)
    """
    lengths    = np.array([float(c.split("=")[1]) for c in row.index])
    capacities = np.array(row.values, dtype=float)

    if L < lengths[0]:
        print(f"  Warning: L={L}m below minimum tabulated {lengths[0]}m. "
              f"Using L={lengths[0]}m")
        return float(capacities[0])

    if L > lengths[-1]:
        raise ValueError(
            f"L={L}m exceeds maximum tabulated length {lengths[-1]}m. "
            f"Table does not extrapolate — check your span."
        )

    f_interp = interp1d(lengths, capacities, kind="linear")
    return float(round(float(f_interp(L)), 1))


# ── Module 4a — Main callable (local Jupyter) ────────────────────────────────
def lookup_ltb_capacity(
    section: str,
    L: float,
    C1: float = 1.0,
    url: str = CSV_URL_S275
) -> dict:
    """
    Main callable for local Jupyter — looks up LTB buckling resistance Mb,Rd.
    Fetches CSV directly from GitHub URL.

    Parameters
    ----------
    section : str
        Section designation e.g. "457 x 191 x 82"
    L : float
        Unrestrained length between restraints (m)
    C1 : float, optional
        Moment gradient factor (default 1.0 — uniform moment, conservative)
    url : str, optional
        Raw GitHub URL to CSV (default S275 table)

    Returns
    -------
    dict
        {
          "section" : str,    # section designation
          "L"       : float,  # unrestrained length (m)
          "C1"      : float,  # moment gradient factor used
          "Mb_Rd"   : float,  # LTB buckling resistance (kNm)
          "steel"   : str,    # steel grade
        }

    Example
    -------
    >>> result = lookup_ltb_capacity("457 x 191 x 82", L=4.5, C1=1.0)
    >>> print(result["Mb_Rd"])
    359.0
    """
    df    = load_ltb_table(url)
    row   = get_ltb_row(df, section, C1)
    Mb_Rd = interpolate_ltb_capacity(row, L)
    steel = "S355" if "S355" in url else "S275"

    return {
        "section" : section,
        "L"       : L,
        "C1"      : C1,
        "Mb_Rd"   : Mb_Rd,
        "steel"   : steel,
    }


# ── Module 4b — Main callable (Claude sandbox) ───────────────────────────────
def lookup_ltb_capacity_from_string(
    csv_text: str,
    section: str,
    L: float,
    C1: float = 1.0,
    steel: str = "S275"
) -> dict:
    """
    Main callable for Claude sandbox — looks up LTB buckling resistance Mb,Rd.
    Accepts raw CSV text string instead of URL (Claude fetches via web_fetch).

    Parameters
    ----------
    csv_text : str
        Raw CSV content fetched via web_fetch
    section : str
        Section designation e.g. "457 x 191 x 82"
    L : float
        Unrestrained length between restraints (m)
    C1 : float, optional
        Moment gradient factor (default 1.0 — uniform moment, conservative)
    steel : str, optional
        Steel grade label for output (default "S275")

    Returns
    -------
    dict
        {
          "section" : str,    # section designation
          "L"       : float,  # unrestrained length (m)
          "C1"      : float,  # moment gradient factor used
          "Mb_Rd"   : float,  # LTB buckling resistance (kNm)
          "steel"   : str,    # steel grade
        }
    """
    df    = load_ltb_table_from_string(csv_text)
    row   = get_ltb_row(df, section, C1)
    Mb_Rd = interpolate_ltb_capacity(row, L)

    return {
        "section" : section,
        "L"       : L,
        "C1"      : C1,
        "Mb_Rd"   : Mb_Rd,
        "steel"   : steel,
    }
