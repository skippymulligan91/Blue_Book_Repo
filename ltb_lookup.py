# ltb_lookup.py
# LTB buckling resistance lookup for UB sections per SCI Blue Book
# Supports S275 (S355 ready via url parameter)
# David Mulligan — Blyth & Blyth

import pandas as pd
import numpy as np
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


# ── Module 1 — Load table ────────────────────────────────────────────────────
def load_ltb_table(url: str = CSV_URL_S275) -> pd.DataFrame:
    """
    Load LTB capacity table from GitHub raw CSV.

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
    df.columns = df.columns.str.strip()  # clean any whitespace in headers
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
        Full LTB table from load_ltb_table()
    section : str
        Section designation e.g. "457 x 191 x 82"
    C1 : float
        Moment gradient factor e.g. 1.0, 1.35 etc.

    Returns
    -------
    pd.Series
        Row of Mb,Rd capacities (kNm) indexed by length e.g. "L=1.0" ... "L=14.0"
    """
    # Filter to requested section
    df_sec = df[df["Section"] == section].copy()

    if df_sec.empty:
        raise ValueError(
            f"Section '{section}' not found in table. "
            f"Check designation format e.g. '457 x 191 x 82'"
        )

    # Available C1 values for this section
    c1_vals = sorted(df_sec["C1"].unique())

    # Capacity columns only
    len_cols = [c for c in df.columns if c.startswith("L=")]

    # Exact C1 match
    if C1 in c1_vals:
        row = df_sec[df_sec["C1"] == C1][len_cols].iloc[0]
        return row

    # C1 below minimum — clamp conservatively
    if C1 < c1_vals[0]:
        print(f"  Warning: C1={C1} below minimum tabulated {c1_vals[0]}. "
              f"Using C1={c1_vals[0]} (conservative)")
        row = df_sec[df_sec["C1"] == c1_vals[0]][len_cols].iloc[0]
        return row

    # C1 above maximum — clamp at top
    if C1 > c1_vals[-1]:
        print(f"  Warning: C1={C1} above maximum tabulated {c1_vals[-1]}. "
              f"Using C1={c1_vals[-1]}")
        row = df_sec[df_sec["C1"] == c1_vals[-1]][len_cols].iloc[0]
        return row

    # Interpolate between two bounding C1 rows
    c1_lo = max(v for v in c1_vals if v < C1)
    c1_hi = min(v for v in c1_vals if v > C1)

    row_lo = df_sec[df_sec["C1"] == c1_lo][len_cols].iloc[0]
    row_hi = df_sec[df_sec["C1"] == c1_hi][len_cols].iloc[0]

    # Linear interpolation factor
    t = (C1 - c1_lo) / (c1_hi - c1_lo)
    row_interp = row_lo + t * (row_hi - row_lo)

    return row_interp


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
    # Extract tabulated lengths and capacities
    lengths    = np.array([float(c.split("=")[1]) for c in row.index])
    capacities = np.array(row.values, dtype=float)

    # Below minimum — clamp conservatively
    if L < lengths[0]:
        print(f"  Warning: L={L}m below minimum tabulated {lengths[0]}m. "
              f"Using L={lengths[0]}m")
        return float(capacities[0])

    # Above maximum — raise error, no extrapolation
    if L > lengths[-1]:
        raise ValueError(
            f"L={L}m exceeds maximum tabulated length {lengths[-1]}m. "
            f"Table does not extrapolate — check your span."
        )

    # scipy linear interpolation
    f_interp = interp1d(lengths, capacities, kind="linear")

    return float(round(float(f_interp(L)), 1))


# ── Module 4 — Main callable ─────────────────────────────────────────────────
def lookup_ltb_capacity(
    section: str,
    L: float,
    C1: float = 1.0,
    url: str = CSV_URL_S275
) -> dict:
    """
    Main callable — looks up LTB buckling resistance Mb,Rd for a UB section.
    Combines table load, C1 interpolation, and length interpolation in one call.

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
          "section"  : str,    # section designation
          "L"        : float,  # unrestrained length (m)
          "C1"       : float,  # moment gradient factor used
          "Mb_Rd"    : float,  # LTB buckling resistance (kNm)
          "steel"    : str,    # steel grade
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

    # Infer steel grade from URL
    if "S355" in url:
        steel = "S355"
    else:
        steel = "S275"

    return {
        "section" : section,
        "L"       : L,
        "C1"      : C1,
        "Mb_Rd"   : Mb_Rd,
        "steel"   : steel,
    }
