import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("Testing imports...")

try:
    from modules.forecasting_models import HAVE_PROPHET, HAVE_LGBM, HAVE_XGBOOST, HAVE_STATSMODELS
    print(f"Prophet: {HAVE_PROPHET}")
    print(f"LightGBM: {HAVE_LGBM}")
    print(f"XGBoost: {HAVE_XGBOOST}")
    print(f"Statsmodels: {HAVE_STATSMODELS}")
    print("Import successful!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
