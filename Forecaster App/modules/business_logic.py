"""
Business logic functions for handling business adjustments, renewals, and model selection.

Contains functions for applying business rules, processing yearly renewals,
and implementing business-aware model selection logic.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st
from .utils import read_any_excel, coerce_month_start
from .ui_components import fy


def process_yearly_renewals(results, diagnostic_messages):
    """
    Process yearly renewals overlay if file is uploaded.
    
    Args:
        results: Dictionary of model results
        diagnostic_messages: List to append diagnostic messages
        
    Returns:
        bool: Whether yearly renewals were applied
    """
    if 'yearly_renewals_file' not in st.session_state or st.session_state['yearly_renewals_file'] is None:
        return False
    
    try:
        yearly_file = st.session_state['yearly_renewals_file']
        yearly_renewals_data = read_any_excel(io.BytesIO(yearly_file.read()))
        
        # Validate yearly renewals data structure
        required_yearly_cols = {"Date", "Product", "ACR"}
        if not required_yearly_cols.issubset(yearly_renewals_data.columns):
            diagnostic_messages.append(f"❌ Yearly Renewals Error: File must contain columns {required_yearly_cols}")
            return False
        
        # Process yearly renewals data
        yearly_renewals_data["Date"] = coerce_month_start(yearly_renewals_data["Date"])
        yearly_renewals_data.sort_values("Date", inplace=True)
        
        # Apply renewals to all models
        _apply_renewals_to_models(results, yearly_renewals_data, diagnostic_messages)
        
        historical_count = len(yearly_renewals_data)
        diagnostic_messages.append(
            f"✅ Non-Compliant Upfront RevRec: Added {historical_count} historical entries + "
            f"projected future renewals at 100% probability (only last 12 months used for projection)"
        )
        
        return True
        
    except Exception as e:
        diagnostic_messages.append(f"❌ Yearly Renewals Error: {str(e)[:100]}")
        return False


def _apply_renewals_to_models(results, yearly_renewals_data, diagnostic_messages):
    """Apply yearly renewals overlay to all models."""
    all_future_entries = 0
    all_future_details = []
    
    for model_name in results.keys():
        if model_name in results and results[model_name] is not None:
            df_model = results[model_name].copy()
            
            # Add yearly renewals as additional rows
            yearly_renewal_rows = []
            
            # Add historical renewals
            for _, renewal_row in yearly_renewals_data.iterrows():
                new_row = {
                    "Product": renewal_row["Product"],
                    "Date": renewal_row["Date"],
                    "ACR": renewal_row["ACR"],
                    "Type": "non-compliant",
                    "FiscalYear": fy(pd.Timestamp(renewal_row["Date"]))
                }
                yearly_renewal_rows.append(new_row)
            
            # Project future renewals
            forecast_dates = df_model[df_model["Type"] == "forecast"]["Date"].unique()
            future_renewals = _project_future_renewals(
                yearly_renewals_data, forecast_dates, model_name, list(results.keys())[0]
            )
            yearly_renewal_rows.extend(future_renewals)
            
            # Track future entries for first model only
            if model_name == list(results.keys())[0]:
                all_future_entries = len(future_renewals)
                all_future_details = [
                    f"{row['Product']} on {pd.Timestamp(row['Date']).strftime('%Y-%m')} = ${row['ACR']:,.0f}"
                    for row in future_renewals[:5]
                ]
            
            if yearly_renewal_rows:
                yearly_df = pd.DataFrame(yearly_renewal_rows)
                df_model = pd.concat([df_model, yearly_df], ignore_index=True)
                df_model = df_model.sort_values(['Product', 'Date']).reset_index(drop=True)
            
            results[model_name] = df_model
    
    # Add future renewals diagnostic message
    if all_future_entries > 0 and all_future_details:
        unique_months = len(set(detail.split(' on ')[1].split(' =')[0] for detail in all_future_details))
        diagnostic_messages.append(
            f"🔮 Future Non-Compliant Upfront RevRec Renewals: {all_future_entries} total projections "
            f"across {unique_months} monthly patterns (based on last 12 months only) - {', '.join(all_future_details)}{'...' if len(all_future_details) > 5 else ''}"
        )


def _project_future_renewals(yearly_renewals_data, forecast_dates, model_name, first_model_name):
    """Project historical renewals into future forecast periods.
    
    Only considers renewals from the last 12 months to avoid over-forecasting
    from potentially churned customers or outdated renewal patterns.
    """
    future_renewals = []
    
    if len(forecast_dates) == 0:
        return future_renewals
    
    forecast_start = pd.Timestamp(forecast_dates.min())
    forecast_end = pd.Timestamp(forecast_dates.max())
    
    # Calculate cutoff date for last 12 months from forecast start
    twelve_months_ago = forecast_start - pd.DateOffset(months=12)
    
    # For each product in renewals data, project the pattern forward
    for product in yearly_renewals_data["Product"].unique():
        product_renewals = yearly_renewals_data[yearly_renewals_data["Product"] == product].copy()
        
        if len(product_renewals) > 0:
            # Filter to only last 12 months of renewals
            product_renewals['Date'] = pd.to_datetime(product_renewals['Date'])
            recent_renewals = product_renewals[product_renewals['Date'] >= twelve_months_ago]
            
            # For each recent renewal (last 12 months), project it forward yearly
            for _, historical_renewal in recent_renewals.iterrows():
                renewal_amount = historical_renewal["ACR"]
                historical_date = pd.Timestamp(historical_renewal["Date"])
                
                # Project this specific renewal forward yearly
                projection_year = historical_date.year + 1
                max_projection_year = forecast_end.year + 1
                
                while projection_year <= max_projection_year:
                    try:
                        next_renewal_date = historical_date.replace(year=projection_year)
                    except ValueError:
                        # Handle leap year edge case (Feb 29 -> Feb 28)
                        next_renewal_date = historical_date.replace(year=projection_year, day=28)
                    
                    # Check if this renewal falls within the forecast period
                    if forecast_start <= next_renewal_date <= forecast_end:
                        future_renewal_row = {
                            "Product": product,
                            "Date": next_renewal_date,
                            "ACR": renewal_amount,
                            "Type": "non-compliant-forecast",
                            "FiscalYear": fy(next_renewal_date)
                        }
                        future_renewals.append(future_renewal_row)
                    
                    projection_year += 1
    
    return future_renewals


def apply_business_adjustments_to_results(results, apply_adjustments, growth_assumption, market_multiplier, market_conditions, diagnostic_messages):
    """
    Apply business adjustments to forecast results if enabled.
    
    Args:
        results: Dictionary of model results
        apply_adjustments: Whether to apply business adjustments
        growth_assumption: Annual growth percentage
        market_multiplier: Market condition multiplier
        market_conditions: Market condition string
        diagnostic_messages: List to append messages
        
    Returns:
        dict: Updated results with business adjustments applied
    """
    if not apply_adjustments:
        return results
    
    from .models import apply_business_adjustments_to_forecast
    
    adjusted_results = {}
    for model_name, model_data in results.items():
        if model_data is not None:
            adjusted_data = model_data.copy()
            
            # Apply adjustments to forecast rows only
            forecast_mask = adjusted_data["Type"] == "forecast"
            if forecast_mask.any():
                forecast_values = adjusted_data.loc[forecast_mask, "ACR"].values
                adjusted_values = apply_business_adjustments_to_forecast(
                    forecast_values, growth_assumption, market_multiplier
                )
                adjusted_data.loc[forecast_mask, "ACR"] = adjusted_values
                
                # Add diagnostic message for first model only
                if model_name == list(results.keys())[0]:
                    diagnostic_messages.append(
                        f"📈 Business Adjustments: Applied {growth_assumption}% growth + "
                        f"{market_conditions} market conditions to all forecasts"
                    )
            
            adjusted_results[model_name] = adjusted_data
        else:
            adjusted_results[model_name] = model_data
    
    return adjusted_results


def create_hybrid_best_model(results, best_models_per_product, best_mapes_per_product, products):
    """
    Create hybrid "Best per Product" model combining best forecasts for each product.
    
    Args:
        results: Dictionary of model results
        best_models_per_product: Dict of best model per product
        best_mapes_per_product: Dict of best MAPE per product
        products: List of product names
        
    Returns:
        DataFrame: Hybrid results or None if creation fails
    """
    hybrid_results = []
    
    for product in products:
        if product in best_models_per_product:
            best_model_name = best_models_per_product[product]
            if best_model_name in results:
                product_data = results[best_model_name]
                product_specific_data = product_data[product_data["Product"] == product].copy()
                if not product_specific_data.empty:
                    product_specific_data["BestModel"] = best_model_name  # Add metadata
                    hybrid_results.append(product_specific_data)
    
    if hybrid_results:
        hybrid_df = pd.concat(hybrid_results, ignore_index=True)
        hybrid_df["FiscalYear"] = hybrid_df["Date"].apply(fy)
        return hybrid_df
    
    return None


def calculate_model_rankings(product_mapes, product_smapes, product_mases, product_rmses, model_names, products):
    """
    Calculate model rankings across multiple metrics.
    
    Args:
        product_mapes, product_smapes, product_mases, product_rmses: Per-product metrics
        model_names: List of model names
        products: List of product names
        
    Returns:
        tuple: (metric_ranks, avg_ranks, best_model_by_rank)
    """
    # Initialize metric ranks
    metric_ranks = {m: {model: 0 for model in model_names} for m in ["MAPE", "SMAPE", "MASE", "RMSE"]}
    
    # For each product, rank models for each metric
    for product in products:
        # MAPE ranking (with safety checks)
        mape_vals = {}
        for model in model_names:
            if model in product_mapes and product_mapes[model] is not None:
                mape_vals[model] = product_mapes[model].get(product, np.nan)
            else:
                mape_vals[model] = np.nan
        mape_sorted = sorted((v, k) for k, v in mape_vals.items() if not np.isnan(v))
        for rank, (v, k) in enumerate(mape_sorted):
            metric_ranks["MAPE"][k] += rank + 1
        
        # SMAPE ranking (with safety checks)
        smape_vals = {}
        for model in model_names:
            if model in product_smapes and product_smapes[model] is not None:
                smape_vals[model] = product_smapes[model].get(product, np.nan)
            else:
                smape_vals[model] = np.nan
        smape_sorted = sorted((v, k) for k, v in smape_vals.items() if not np.isnan(v))
        for rank, (v, k) in enumerate(smape_sorted):
            metric_ranks["SMAPE"][k] += rank + 1
        
        # MASE ranking (with safety checks)
        mase_vals = {}
        for model in model_names:
            if model in product_mases and product_mases[model] is not None:
                mase_vals[model] = product_mases[model].get(product, np.nan)
            else:
                mase_vals[model] = np.nan
        mase_sorted = sorted((v, k) for k, v in mase_vals.items() if not np.isnan(v))
        for rank, (v, k) in enumerate(mase_sorted):
            metric_ranks["MASE"][k] += rank + 1
        
        # RMSE ranking (with safety checks)
        rmse_vals = {}
        for model in model_names:
            if model in product_rmses and product_rmses[model] is not None:
                rmse_vals[model] = product_rmses[model].get(product, np.nan)
            else:
                rmse_vals[model] = np.nan
        rmse_sorted = sorted((v, k) for k, v in rmse_vals.items() if not np.isnan(v))
        for rank, (v, k) in enumerate(rmse_sorted):
            metric_ranks["RMSE"][k] += rank + 1
    
    # Compute average rank for each model
    avg_ranks = {model: np.mean([metric_ranks[m][model] for m in metric_ranks]) for model in model_names}
    
    # Select best model by average rank
    best_model_by_rank = min(avg_ranks.keys(), key=lambda k: avg_ranks[k]) if avg_ranks else None
    
    return metric_ranks, avg_ranks, best_model_by_rank


def find_best_models_per_product(results, product_mapes, product_smapes, product_mases, product_rmses, products, enable_business_aware_selection, diagnostic_messages):
    """
    Find the best model for each product using multi-metric ranking.
    
    Args:
        results: Dictionary of model results
        product_mapes, product_smapes, product_mases, product_rmses: Per-product metrics
        products: List of product names
        enable_business_aware_selection: Whether to use business-aware selection
        diagnostic_messages: List to append messages
        
    Returns:
        tuple: (best_models_per_product, best_mapes_per_product)
    """
    from .models import select_business_aware_best_model
    
    best_models_per_product = {}
    best_mapes_per_product = {}
    model_names = list(results.keys())
    
    for product in products:
        # Calculate per-product rankings across all 4 metrics
        product_model_metrics = {}
        product_model_ranks = {model: 0 for model in model_names}
        
        for model_name in results.keys():
            if product in product_mapes[model_name]:
                product_model_metrics[model_name] = {
                    'MAPE': product_mapes[model_name][product],
                    'SMAPE': product_smapes[model_name].get(product, np.nan),
                    'MASE': product_mases[model_name].get(product, np.nan),
                    'RMSE': product_rmses[model_name].get(product, np.nan)
                }
        
        if product_model_metrics:
            # Calculate rankings for this specific product across all metrics
            for metric in ['MAPE', 'SMAPE', 'MASE', 'RMSE']:
                metric_vals = {model: metrics[metric] for model, metrics in product_model_metrics.items() 
                             if not np.isnan(metrics[metric])}
                if metric_vals:
                    metric_sorted = sorted(metric_vals.items(), key=lambda x: x[1])
                    for rank, (model, _) in enumerate(metric_sorted):
                        product_model_ranks[model] += rank + 1
            
            # Select model with best average rank for this product
            if enable_business_aware_selection:
                # Apply business-aware filtering to ranked candidates
                product_model_mapes = {model: product_mapes[model][product] 
                                     for model in product_model_metrics.keys()}
                # Pass ranking information for smarter business-aware selection
                valid_ranks = {model: rank/4 for model, rank in product_model_ranks.items() if rank > 0}
                best_model_for_product, best_mape_for_product = select_business_aware_best_model(
                    product_model_mapes, product, diagnostic_messages, valid_ranks
                )
            else:
                # Pure multi-metric ranking selection
                valid_ranks = {model: rank for model, rank in product_model_ranks.items() if rank > 0}
                if valid_ranks:
                    best_model_for_product = min(valid_ranks.keys(), key=lambda k: valid_ranks[k])
                    best_mape_for_product = product_mapes[best_model_for_product][product]
                    
                    if diagnostic_messages:
                        avg_rank = valid_ranks[best_model_for_product] / 4
                        diagnostic_messages.append(
                            f"📊 Product {product}: Multi-metric ranking selected {best_model_for_product} "
                            f"(Avg Rank: {avg_rank:.1f}, MAPE: {best_mape_for_product:.1%})"
                        )
                else:
                    # Fallback to MAPE if ranking fails
                    product_model_mapes = {model: product_mapes[model][product] 
                                         for model in product_model_metrics.keys()}
                    best_model_for_product = min(product_model_mapes.keys(), key=lambda k: product_model_mapes[k])
                    best_mape_for_product = product_model_mapes[best_model_for_product]
                
            best_models_per_product[product] = best_model_for_product
            best_mapes_per_product[product] = best_mape_for_product
        else:
            # Fallback to first available model
            best_models_per_product[product] = list(results.keys())[0]
            best_mapes_per_product[product] = 1.0
    
    return best_models_per_product, best_mapes_per_product
