# Version 1.4.0 Release Summary

**Release Date**: November 13, 2025  
**Status**: 🚀 Ready for Tagging and GitHub Release

## 📦 What Is Included

### Version 1.4.0 Highlights
- **Per-product Fiscal-Year Totals Everywhere**: Results dashboard and adjustment expanders now surface combined actual, forecast, and non-compliant revenue totals for each product.
- **Boundary-Aware Fiscal Year Smoothing**: Optional toggle blends July handoffs to tame spikes when user overrides diverge from baseline models.
- **Streamlined Model Review & Exports**: Model selector sits atop the chart and exports use the compact `Base_Model_YYYYMMDD_HHMMSS.xlsx` pattern to avoid Windows path limits.

## 🚀 GitHub Release Preparation

### Repository Information
- **Repository**: https://github.com/jakemoura/Forecasting_App
- **Branch**: main
- **Planned Tag**: v1.4.0 (to be created after commit)

### Files Updated for 1.4.0
1. **CHANGELOG.md**
   - Added v1.4.0 entry, version history row, and upgrade guidance.
2. **VERSION.txt**
   - Bumped semantic version to `1.4.0`.
3. **RELEASE_NOTES_v1.4.0.md** (new)
   - Detailed release notes used for GitHub release description.
4. **RELEASE_v1.4.0_SUMMARY.md** (this document)
   - Snapshot of deliverables, release steps, and verification checklist.

## ✅ Verification Checklist

- [x] Code/UI changes merged into `main`
- [x] Version metadata bumped to 1.4.0
- [x] Changelog updated with 1.4.0 details
- [x] Release notes drafted (`RELEASE_NOTES_v1.4.0.md`)
- [x] Release summary prepared (this file)
- [ ] Tag `v1.4.0` created locally
- [ ] Commit and tag pushed to GitHub
- [ ] GitHub release published with release notes

## 📄 GitHub Release Instructions

1. Commit staged changes: `git commit -m "docs(release): finalize 1.4.0 launch"`
2. Create tag: `git tag v1.4.0`
3. Push branch and tag: `git push origin main --tags`
4. Publish GitHub release at https://github.com/jakemoura/Forecasting_App/releases/new
   - Tag: `v1.4.0`
   - Release title: `Version 1.4.0 - Fiscal-Year Totals Everywhere`
   - Body: paste from `RELEASE_NOTES_v1.4.0.md`
   - Set as latest release and publish

## 📊 Quick Stats

- **Impacted Modules**: `modules/ui_components.py`
- **New Documentation Files**: 2 (release notes + summary)
- **Backward Compatibility**: Maintained; no migrations required
- **Python Compatibility**: 3.11 / 3.12 recommended; 3.13 requires latest wheels

---

*Prepared on November 13, 2025 for the Forecasting App v1.4.0 release.*
