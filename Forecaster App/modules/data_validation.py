"""
Data validation and quality analysis functions.

Contains functions for validating uploaded data, analyzing data quality,
and providing feedback on data suitability for forecasting.
"""

import pandas as pd
import numpy as np
import streamlit as st
from .utils import coerce_month_start
from .models import detect_seasonality_strength, get_seasonality_aware_split


def validate_data_format(raw_data):
    """
    Validate that the uploaded data has required columns.
    
    Args:
        raw_data: DataFrame with uploaded data
        
    Returns:
        bool: True if data format is valid
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = {"Date", "Product", "ACR"}
    if not required_columns.issubset(raw_data.columns):
        raise ValueError(f"Workbook must contain columns: {required_columns}")
    return True


def prepare_data(raw_data):
    """
    Prepare and clean the uploaded data for analysis.
    
    Args:
        raw_data: DataFrame with uploaded data
        
    Returns:
        DataFrame: Cleaned and prepared data
        
    Raises:
        ValueError: If date parsing fails
    """
    try:
        raw_data["Date"] = coerce_month_start(raw_data["Date"])
        raw_data.sort_values("Date", inplace=True)
        return raw_data
    except ValueError as e:
        raise ValueError(f"Date format error: {str(e)}")


def analyze_data_quality(raw_data):
    """
    Analyze data quality for each product and return analysis results.
    
    Args:
        raw_data: Prepared DataFrame with Date, Product, ACR columns
        
    Returns:
        tuple: (data_analysis_list, overall_status)
    """
    data_analysis = []
    overall_status = "good"
    
    for product, grp in raw_data.groupby("Product"):
        try:
            series = grp.set_index("Date")["ACR"].astype(float)
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)
            series.index = series.index.to_period("M").to_timestamp(how="start")
            
            months_count = len(series)
            date_range = f"{series.index.min().strftime('%b %Y')} to {series.index.max().strftime('%b %Y')}"
            
            # Determine status with indicators
            status_info = _get_data_status(months_count)
            
            # Update overall status
            if status_info['status_level'] == 'insufficient':
                overall_status = "insufficient"
            elif status_info['status_level'] == 'limited' and overall_status != "insufficient":
                overall_status = "limited"
            elif status_info['status_level'] == 'moderate' and overall_status in ["good"]:
                overall_status = "moderate"
            
            data_analysis.append({
                "Product": product,
                "📊 Months": f"{months_count} months",
                "📅 Date Range": date_range,
                "🎯 Status": f"{status_info['icon']} {status_info['text']}",
                "📈 Expected MAPE": status_info['accuracy'],
                "💡 Recommendation": status_info['recommendation']
            })
            
        except Exception as e:
            data_analysis.append({
                "Product": product,
                "📊 Months": "Error",
                "📅 Date Range": "Invalid data",
                "🎯 Status": "🔴 Error",
                "📈 Expected MAPE": "N/A",
                "💡 Recommendation": f"Data error: {str(e)[:30]}..."
            })
            overall_status = "insufficient"
    
    return data_analysis, overall_status


def _get_data_status(months_count):
    """
    Get status information based on months count.
    
    Args:
        months_count: Number of months in the data
        
    Returns:
        dict: Status information including icon, text, accuracy, etc.
    """
    if months_count >= 36:
        return {
            'icon': "🟢",
            'text': "Excellent",
            'accuracy': "5-15%",
            'recommendation': "All models optimal",
            'status_level': 'excellent'
        }
    elif months_count >= 24:
        return {
            'icon': "🟡",
            'text': "Good",
            'accuracy': "10-20%",
            'recommendation': "Most models work well",
            'status_level': 'moderate'
        }
    elif months_count >= 18:
        return {
            'icon': "🟠",
            'text': "Moderate",
            'accuracy': "15-30%",
            'recommendation': "Polynomial models recommended",
            'status_level': 'limited'
        }
    else:
        return {
            'icon': "🔴",
            'text': "Insufficient",
            'accuracy': "20%+",
            'recommendation': "Need more data",
            'status_level': 'insufficient'
        }


def display_data_analysis_results(data_analysis, overall_status):
    """
    Display the data analysis results in a formatted table and overall status.
    
    Args:
        data_analysis: List of data analysis results
        overall_status: Overall data quality status
    """
    st.markdown("---")
    st.markdown("### 📊 **Data Analysis Results**")
    
    # Display results table
    if data_analysis:
        analysis_df = pd.DataFrame(data_analysis)
        st.dataframe(
            analysis_df, 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "🎯 Status": st.column_config.TextColumn(width="medium"),
                "📈 Expected MAPE": st.column_config.TextColumn(width="small"),
                "💡 Recommendation": st.column_config.TextColumn(width="large")
            }
        )
    
    # Display overall recommendation
    st.markdown("<br>", unsafe_allow_html=True)
    _display_overall_status(overall_status)
    st.markdown("---")


def _display_overall_status(overall_status):
    """Display the overall data quality status message."""
    if overall_status == "good":
        st.success("🎉 **Excellent Data Quality** - Ready for high-accuracy forecasting!")
    elif overall_status == "moderate":
        st.info("✅ **Good Data Quality** - Should produce reliable forecasts with most models")
    elif overall_status == "limited":
        st.warning("⚠️ **Limited Data** - Basic forecasting possible. Consider Polynomial models for best results")
    else:
        st.error("❌ **Insufficient Data** - Need at least 18 months for reliable forecasting")
        st.stop()


def display_date_format_error(error_message):
    """
    Display date format error with helpful guidance.
    
    Args:
        error_message: The error message to display
    """
    st.error(f"❌ **Date Format Error**")
    st.markdown(f"**Issue:** {error_message}")
    
    with st.expander("📖 **Supported Date Formats**", expanded=True):
        st.markdown("""
        **Examples of supported formats:**
        - `Apr-22`, `May-22` (interpreted as 2022)
        - `Apr-2022`, `May-2022`  
        - `April-22`, `May-22`
        - `04/2022`, `05/2022`
        - Standard Excel date formats
        """)
    st.stop()


def get_valid_products(raw_data, diagnostic_messages=None):
    """
    Get list of products that have sufficient data for forecasting.
    
    Args:
        raw_data: Prepared DataFrame
        diagnostic_messages: Optional list to append diagnostic messages
        
    Returns:
        list: Valid product names that can be forecasted
    """
    valid_products = []
    
    for product, grp in raw_data.groupby("Product"):
        try:
            series = grp.set_index("Date")["ACR"].astype(float)
            if not isinstance(series.index, pd.DatetimeIndex):
                try:
                    series.index = pd.to_datetime(series.index)
                except (ValueError, pd.errors.OutOfBoundsDatetime) as e:
                    if diagnostic_messages is not None:
                        diagnostic_messages.append(f"❌ Product {product}: Date parsing error - {str(e)[:100]}. Skipping.")
                    continue
            series.index = series.index.to_period("M").to_timestamp(how="start")
        except Exception as e:
            if diagnostic_messages is not None:
                diagnostic_messages.append(f"❌ Product {product}: Data processing error - {str(e)[:100]}. Skipping.")
            continue
        
        # Check if product has sufficient data
        if len(series) < 12:
            continue
        
        # Check seasonality-aware split
        seasonality_strength = detect_seasonality_strength(series)
        split_result = get_seasonality_aware_split(series, seasonal_period=12, diagnostic_messages=None)
        if split_result[0] is not None:
            valid_products.append(product)
    
    return valid_products
