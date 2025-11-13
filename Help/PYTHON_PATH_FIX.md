# CRITICAL: Python Not Found in PATH

## 🚨 THE PROBLEM

Python is installed but **NOT in your system PATH**, so Windows can't find it when you run commands.

This is why SETUP.bat fails immediately.

## ✅ SOLUTION 1: Reinstall Python with PATH (RECOMMENDED)

### Step 1: Uninstall Current Python
1. Press **Windows Key**
2. Type "Add or remove programs"
3. Find "Python 3.14" or "Python 3.13"
4. Click **Uninstall**
5. Follow the prompts

### Step 2: Install Python 3.12 (RECOMMENDED)

#### Option A: Microsoft Store (EASIEST - Automatically adds to PATH)
1. Press **Windows Key**
2. Type "Microsoft Store" and open it
3. Search for **"Python 3.12"**
4. Click **Get** / **Install**
5. Wait for installation
6. **Done!** - PATH is automatically configured

#### Option B: Python.org
1. Visit: https://www.python.org/downloads/
2. Download **Python 3.12.x** (latest 3.12 version)
3. **RUN THE INSTALLER**
4. ⚠️ **CRITICAL: CHECK "Add python.exe to PATH"** checkbox at the bottom
5. Click **"Install Now"**
6. Wait for installation
7. Click **Close**

### Step 3: Verify Installation
1. **Close all PowerShell/CMD windows**
2. Open **NEW** PowerShell
3. Run: `python --version`
4. Should show: `Python 3.12.x`

### Step 4: Run Setup
```batch
cd "c:\Users\jakemoura\OneDrive - Microsoft\Desktop\Repos\Forecasting_App-1"
.\SETUP.bat
```

---

## ✅ SOLUTION 2: Add Existing Python to PATH (If you want to keep current version)

### Step 1: Find Python Installation Location

#### Option A: Using Windows Search
1. Press **Windows Key**
2. Search for "python.exe"
3. Right-click on **Python 3.xx**
4. Click **"Open file location"**
5. Note the path (e.g., `C:\Users\jakemoura\AppData\Local\Programs\Python\Python314`)

#### Option B: Common Locations
Check these folders:
- `C:\Users\jakemoura\AppData\Local\Programs\Python\Python314`
- `C:\Users\jakemoura\AppData\Local\Programs\Python\Python313`
- `C:\Users\jakemoura\AppData\Local\Microsoft\WindowsApps\`
- `C:\Python314`
- `C:\Program Files\Python314`

### Step 2: Add to PATH

1. Press **Windows Key**
2. Type "environment variables"
3. Click **"Edit the system environment variables"**
4. Click **"Environment Variables"** button
5. In **"User variables"** section, find and select **"Path"**
6. Click **"Edit"**
7. Click **"New"**
8. Add the Python folder path (e.g., `C:\Users\jakemoura\AppData\Local\Programs\Python\Python314`)
9. Click **"New"** again
10. Add the Scripts folder path (e.g., `C:\Users\jakemoura\AppData\Local\Programs\Python\Python314\Scripts`)
11. Click **OK** on all windows

### Step 3: Restart Terminal
1. **Close ALL PowerShell/CMD windows**
2. Open **NEW** PowerShell
3. Test: `python --version`

---

## 🎯 WHICH SOLUTION TO USE?

### Use Solution 1 (Reinstall) if:
- ✅ You have Python 3.14 or 3.13 (not ideal for data science)
- ✅ You want the easiest fix
- ✅ You want Python 3.12 (RECOMMENDED for this project)

### Use Solution 2 (Add to PATH) if:
- You specifically need Python 3.14 for another project
- You don't want to reinstall

---

## 🚀 AFTER PYTHON IS IN PATH

Once `python --version` works, run:

```batch
cd "c:\Users\jakemoura\OneDrive - Microsoft\Desktop\Repos\Forecasting_App-1"
.\CHECK_PYTHON_VERSION.bat
```

This will:
- Show your Python version
- Tell you if it's compatible
- Recommend next steps

Then run:
```batch
.\SETUP.bat
```

---

## 💡 WHY MICROSOFT STORE PYTHON IS BEST

Microsoft Store version automatically:
- ✅ Adds to PATH
- ✅ Updates automatically
- ✅ Works immediately
- ✅ No configuration needed

**RECOMMENDED**: Uninstall current Python, install Python 3.12 from Microsoft Store!

---

## 🔍 QUICK TEST COMMANDS

After fixing PATH, test with:

```powershell
# Test Python is found
python --version

# Test pip works
python -m pip --version

# Test package installation
python -m pip install --upgrade pip

# Check installed location
python -c "import sys; print(sys.executable)"
```

All should work without errors!

---

## ❓ STILL NOT WORKING?

If after reinstalling Python 3.12 from Microsoft Store it still doesn't work:

1. **Restart your computer** (sometimes needed for PATH changes)
2. **Open NEW PowerShell** (must be new window after restart)
3. Try: `python --version`

If STILL not working:
- Make sure you closed ALL old terminals
- Check Windows Store shows Python 3.12 as "Installed"
- Try: `py --version` (Windows launcher)
- Try: `py -3.12 --version`

---

## 📋 SUMMARY

**The Issue**: Python not in PATH

**Best Fix**: Install Python 3.12 from Microsoft Store (automatically adds to PATH)

**After Fix**: Run `.\SETUP.bat` to install all packages

**Time**: 5 minutes to fix, 3-5 minutes for setup = **8-10 minutes total** to working app!
