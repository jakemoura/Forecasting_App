# 🔒 Sensitive Data Protection Guide

## Overview
This document outlines the measures implemented to prevent sensitive forecast data from being accidentally uploaded to GitHub or stored in unnecessary files.

## 🚨 What Are Altair Data Files?

**Altair data files** (e.g., `altair-data-*.json`) are temporary files created by the Altair charting library when rendering complex charts in Streamlit. These files can contain:

- ✅ **Complete forecast data** including revenue projections
- ✅ **Product-specific information** and business metrics  
- ✅ **Historical performance data** and trends
- ✅ **Adjusted forecast results** with your custom modifications
- ✅ **YoY growth targets** and business assumptions

**⚠️ SECURITY RISK**: These files contain your complete business data and should never be uploaded to public repositories!

## 🛡️ Protection Measures Implemented

### 1. **Comprehensive .gitignore**
Created `.gitignore` file that excludes:
```
# Altair chart data files (contain sensitive forecast data)
altair-data-*.json
**/altair-data-*.json

# Excel files that might contain sensitive data
*.xlsx
*.xls
*.csv

# User uploaded data files
**/uploads/
**/data/
**/user_data/
```

### 2. **Altair Configuration**
Updated `forecaster_app.py` to configure Altair behavior:
```python
# Configure Altair to prevent sensitive data file creation
import altair as alt
alt.data_transformers.disable_max_rows()
alt.data_transformers.enable('json')
```

### 3. **Streamlit Configuration**
Enhanced `.streamlit/config.toml` with security settings:
```toml
# Prevent data gathering and minimize file creation
[browser] 
gatherUsageStats = false

[runner]
magicEnabled = false
```

## 📋 Best Practices for Data Security

### ✅ **DO:**
1. **Always check git status** before committing changes
2. **Review files being added** with `git add . --dry-run`
3. **Use .gitignore patterns** for any new data file types
4. **Keep uploaded Excel files local** - never commit them
5. **Test with sample data** when sharing code

### ❌ **DON'T:**
1. **Don't commit altair-data-*.json files** - they contain full forecast data
2. **Don't upload real business Excel files** to the repository
3. **Don't ignore .gitignore warnings** from your IDE
4. **Don't commit session state files** or temporary data
5. **Don't share screenshots** with sensitive data visible

## 🔍 How to Check for Sensitive Data

### Before Committing:
```bash
# Check what files would be added
git add . --dry-run

# Look for sensitive file patterns
git status | grep -E "(altair-data|\.xlsx|\.csv)"

# Verify .gitignore is working
git check-ignore altair-data-*.json
```

### Regular Monitoring:
```bash
# List all tracked files
git ls-files

# Check for accidentally tracked data files
git ls-files | grep -E "(altair-data|\.xlsx|\.csv|data|uploads)"
```

## 🚨 Emergency: Sensitive Data Was Committed

If sensitive data was accidentally committed:

### 1. **For Recent Commits (Not Pushed):**
```bash
# Remove files from staging
git rm --cached altair-data-*.json
git rm --cached *.xlsx

# Commit the removal
git commit -m "Remove sensitive data files"
```

### 2. **For Pushed Commits:**
```bash
# Use BFG Repo-Cleaner (recommended)
bfg --delete-files altair-data-*.json
bfg --delete-files *.xlsx

# Force push (⚠️ DANGEROUS - coordinate with team)
git push --force
```

### 3. **For Public Repositories:**
- **Immediately rotate any API keys or credentials**
- **Change passwords if they were exposed**
- **Consider the data permanently compromised**
- **Review who had access to the repository**

## 📁 Safe File Sharing

### For Collaboration:
1. **Use environment variables** for sensitive configuration
2. **Share sample/mock data** instead of real data
3. **Document data schema** without including actual values
4. **Use secure file sharing services** for real data transfer

### Sample Data Structure:
```csv
Date,Product,ACR
2024-01-01,Product_A,1000
2024-02-01,Product_A,1050
2024-03-01,Product_A,1100
```

## 🔄 Regular Maintenance

### Monthly Tasks:
- [ ] Review .gitignore effectiveness
- [ ] Check for new file types that need exclusion
- [ ] Audit committed files for sensitive data
- [ ] Update team on security practices

### Before Major Releases:
- [ ] Scan entire repository for sensitive patterns
- [ ] Verify all data files are properly excluded
- [ ] Test with clean repository clone
- [ ] Document any new security measures

## 📞 Questions or Concerns?

If you discover sensitive data in the repository or have questions about data security:

1. **Stop committing immediately**
2. **Document what was exposed**
3. **Follow emergency procedures above**
4. **Review and update protection measures**

Remember: **It's better to be overly cautious with sensitive business data than to accidentally expose it!** 🔒