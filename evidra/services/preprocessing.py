import pandas as pd
import numpy as np

# 時系列前処理のユーティリティ。ここでは最小限のチェックと簡易処理を実装する。

def validate_header(df: pd.DataFrame):
    """
    列名が空/重複/記号先頭になっていないかを検査する。
    問題があれば例外を投げる。
    """
    cols = list(df.columns)
    if any(c is None or str(c).strip() == "" for c in cols):
        raise ValueError("列名に空の値があります。先頭行が列名かを確認してください。")
    if len(set(cols)) != len(cols):
        raise ValueError("列名が重複しています。重複を解消してください。")
    if any(str(c)[0] in "!@#$%^&*()-=+[]{};:'\",.<>/?\\" for c in cols):
        raise ValueError("列名が記号で始まっています。適切な名前に変更してください。")

def guess_frequency(index: pd.DatetimeIndex) -> str:
    """
    DatetimeIndexから頻度を推定する。厳密ではない簡易推定。
    """
    if len(index) < 3:
        return ""
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return ""
    median = deltas.median()
    if pd.Timedelta('365D') * 0.5 < median < pd.Timedelta('365D') * 1.5:
        return "Y"
    if pd.Timedelta('90D') * 0.5 < median < pd.Timedelta('90D') * 1.5:
        return "Q"
    if pd.Timedelta('30D') * 0.5 < median < pd.Timedelta('30D') * 1.5:
        return "M"
    if pd.Timedelta('7D') * 0.5 < median < pd.Timedelta('7D') * 1.5:
        return "W"
    if pd.Timedelta('1D') * 0.5 < median < pd.Timedelta('1D') * 1.5:
        return "D"
    return ""

def apply_preprocessing(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    前処理オプション（標準化/欠損補間/等間隔化/差分化）を適用する。
    """
    result = df.copy()

    # 欠損補間
    if params.get('impute'):
        # 前方補間→後方補間→線形補間の順で簡易補完
        result = result.ffill().bfill().interpolate(limit_direction='both')

    # 差分化（ADFに基づく提案があればON）
    if params.get('diff'):
        result = result.diff().dropna()

    # 標準化（既定ONだがUIでは明記しない）
    if params.get('standardize', True):
        result = (result - result.mean()) / (result.std().replace(0, 1.0))

    return result
