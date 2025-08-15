; filepath: forecasting_apps_setup_auto.iss
; Inno Setup Script for Forecasting Apps Suite with Automatic Dependency Installation
; Created by GitHub Copilot

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
AppId={{8B2C5D4E-1A7F-4E8B-9C3D-2F5A6B7C8D9E}
AppName=Forecasting Apps Suite
AppVersion=1.0.0
AppVerName=Forecasting Apps Suite 1.0.0
AppPublisher=CoreAI Finance Team
AppPublisherURL=https://github.com/your-org/forecasting-apps
AppSupportURL=https://github.com/your-org/forecasting-apps/issues
AppUpdatesURL=https://github.com/your-org/forecasting-apps/releases
DefaultDirName={autopf}\Forecasting Apps Suite
DefaultGroupName=Forecasting Apps Suite
AllowNoIcons=yes
LicenseFile=LICENSE.txt
InfoBeforeFile=README.txt
InfoAfterFile=SETUP_GUIDE.html
OutputDir=Installer_Output
OutputBaseFilename=Forecasting_Apps_Suite_Setup_Auto
SetupIconFile=forecasting_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\forecasting_icon.ico
UninstallDisplayName=Forecasting Apps Suite
VersionInfoVersion=1.0.0.0
VersionInfoCompany=CoreAI Finance Team
VersionInfoDescription=Advanced Time Series Forecasting Applications with Auto-Setup
VersionInfoCopyright=Copyright (C) 2025 CoreAI Finance Team

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full"; Description: "Full installation (Python dependencies installed manually)"
Name: "compact"; Description: "Compact installation (minimal files only)"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Core Application Files"; Types: full compact custom; Flags: fixed
Name: "mainapp"; Description: "Main Forecasting App"; Types: full compact custom; Flags: fixed
Name: "outlookapp"; Description: "Quarterly Outlook Forecaster"; Types: full compact custom; Flags: fixed
Name: "help"; Description: "Help and Documentation"; Types: full custom
Name: "shortcuts"; Description: "Desktop Shortcuts"; Types: full custom
Name: "autodeps"; Description: "Automatically install Python dependencies (shows progress window, may take 10-15 minutes)"; Types: custom

[Tasks]
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
; Core application files
Source: "Forecaster App\*"; DestDir: "{app}\Forecaster App"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: core mainapp
Source: "Quarter Outlook App\*"; DestDir: "{app}\Quarter Outlook App"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: core outlookapp
Source: "Help\*"; DestDir: "{app}\Help"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: help

; Batch files and configuration
Source: "SETUP.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "SETUP_INSTALLER.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "FIX_PMDARIMA.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "TEST_PMDARIMA.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "FORCE_INSTALL_PMDARIMA.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "INSTALL_PMDARIMA_PYTHON311.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "DIAGNOSE_PMDARIMA.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "RUN_FORECAST_APP.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: mainapp
Source: "RUN_OUTLOOK_FORECASTER.bat"; DestDir: "{app}"; Flags: ignoreversion; Components: outlookapp

; Documentation
Source: "README.txt"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "SETUP_GUIDE.html"; DestDir: "{app}"; Flags: ignoreversion; Components: core
Source: "requirements_portable.txt"; DestDir: "{app}"; Flags: ignoreversion; Components: core

; Icons (install custom icon)
Source: "forecasting_icon.ico"; DestDir: "{app}"; Flags: ignoreversion; Components: core

; Streamlit configuration files to skip email prompts
Source: ".streamlit\config.toml"; DestDir: "{app}\.streamlit"; Flags: ignoreversion; Components: core
Source: ".streamlit\credentials.toml"; DestDir: "{app}\.streamlit"; Flags: ignoreversion; Components: core
Source: "Forecaster App\.streamlit\config.toml"; DestDir: "{app}\Forecaster App\.streamlit"; Flags: ignoreversion; Components: mainapp
Source: "Forecaster App\.streamlit\credentials.toml"; DestDir: "{app}\Forecaster App\.streamlit"; Flags: ignoreversion; Components: mainapp
Source: "Quarter Outlook App\.streamlit\config.toml"; DestDir: "{app}\Quarter Outlook App\.streamlit"; Flags: ignoreversion; Components: outlookapp
Source: "Quarter Outlook App\.streamlit\credentials.toml"; DestDir: "{app}\Quarter Outlook App\.streamlit"; Flags: ignoreversion; Components: outlookapp

[Icons]
; Start Menu shortcuts
Name: "{group}\Main Forecasting App"; Filename: "{app}\RUN_FORECAST_APP.bat"; WorkingDir: "{app}"; Comment: "Launch the Main Forecasting Application"; IconFilename: "{app}\forecasting_icon.ico"; Components: mainapp
Name: "{group}\Quarterly Outlook Forecaster"; Filename: "{app}\RUN_OUTLOOK_FORECASTER.bat"; WorkingDir: "{app}"; Comment: "Launch the Quarterly Outlook Forecaster"; IconFilename: "{app}\forecasting_icon.ico"; Components: outlookapp
Name: "{group}\Setup Python Dependencies"; Filename: "{app}\SETUP.bat"; WorkingDir: "{app}"; Comment: "Install required Python packages"; IconFilename: "{app}\forecasting_icon.ico"; Components: core
Name: "{group}\Fix PMDARIMA Compatibility"; Filename: "{app}\FIX_PMDARIMA.bat"; WorkingDir: "{app}"; Comment: "Fix scipy compatibility issues with pmdarima"; IconFilename: "{app}\forecasting_icon.ico"; Components: core
Name: "{group}\Force Install PMDARIMA"; Filename: "{app}\FORCE_INSTALL_PMDARIMA.bat"; WorkingDir: "{app}"; Comment: "Force reinstall pmdarima with compatible versions"; IconFilename: "{app}\forecasting_icon.ico"; Components: core
Name: "{group}\Install PMDARIMA (Python 3.11+)"; Filename: "{app}\INSTALL_PMDARIMA_PYTHON311.bat"; WorkingDir: "{app}"; Comment: "Install pmdarima for Python 3.11+ using pre-built wheels"; IconFilename: "{app}\forecasting_icon.ico"; Components: core
Name: "{group}\Diagnose PMDARIMA Issues"; Filename: "{app}\DIAGNOSE_PMDARIMA.bat"; WorkingDir: "{app}"; Comment: "Check pmdarima installation and compatibility"; IconFilename: "{app}\forecasting_icon.ico"; Components: core
Name: "{group}\Help and Documentation"; Filename: "{app}\SETUP_GUIDE.html"; Comment: "View setup guide and documentation"; Components: help
Name: "{group}\Application Folder"; Filename: "{app}"; Comment: "Open application folder"
Name: "{group}\{cm:UninstallProgram,Forecasting Apps Suite}"; Filename: "{uninstallexe}"; Comment: "Uninstall Forecasting Apps Suite"

; Desktop shortcuts (created by default with custom icons)
Name: "{autodesktop}\Main Forecasting App"; Filename: "{app}\RUN_FORECAST_APP.bat"; WorkingDir: "{app}"; Comment: "Launch the Main Forecasting Application"; IconFilename: "{app}\forecasting_icon.ico"
Name: "{autodesktop}\Quarterly Outlook Forecaster"; Filename: "{app}\RUN_OUTLOOK_FORECASTER.bat"; WorkingDir: "{app}"; Comment: "Launch the Quarterly Outlook Forecaster"; IconFilename: "{app}\forecasting_icon.ico"

; Quick Launch shortcuts
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\Main Forecasting App"; Filename: "{app}\RUN_FORECAST_APP.bat"; WorkingDir: "{app}"; IconFilename: "{app}\forecasting_icon.ico"; Tasks: quicklaunchicon; Components: mainapp

[Run]
; Automatically run Python dependency setup if component is selected (shows command window)
Filename: "{app}\SETUP_INSTALLER.bat"; WorkingDir: "{app}"; StatusMsg: "Installing Python dependencies (this may take 10-15 minutes)..."; Flags: waituntilterminated; Description: "Install Python dependencies automatically"; Components: autodeps; Check: IsPythonInstalled
; Optionally launch the main app after installation
Filename: "{app}\RUN_FORECAST_APP.bat"; Description: "Launch Main Forecasting App"; Flags: postinstall nowait shellexec skipifsilent; Components: mainapp

[UninstallDelete]
Type: files; Name: "{app}\.streamlit\*"
Type: dirifempty; Name: "{app}\.streamlit"
Type: files; Name: "{app}\Forecaster App\.streamlit\*"
Type: dirifempty; Name: "{app}\Forecaster App\.streamlit"
Type: files; Name: "{app}\Quarter Outlook App\.streamlit\*"
Type: dirifempty; Name: "{app}\Quarter Outlook App\.streamlit"

[Code]
// Custom Pascal Script functions for advanced installer behavior

function IsPythonInstalled(): Boolean;
var
  ResultCode: Integer;
begin
  // Check if Python is available in PATH
  Result := Exec('cmd.exe', '/c python --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  if not Result or (ResultCode <> 0) then
  begin
    // Python not found or error occurred
    Result := False;
  end
  else
  begin
    Result := True;
  end;
end;

function InitializeSetup(): Boolean;
var
  PythonAvailable: Boolean;
  Response: Integer;
begin
  Result := True;
  
  // Check if Python is installed
  PythonAvailable := IsPythonInstalled();
  
  if not PythonAvailable then
  begin
    Response := MsgBox('Python is not installed on your system. The Forecasting Apps require Python to run.' + #13#10 + #13#10 + 
              'Installation options:' + #13#10 +
              '• Continue installation and install Python manually later' + #13#10 +
              '• Cancel and install Python first (recommended)' + #13#10 + #13#10 +
              'Recommended Python sources:' + #13#10 +
              '• Microsoft Store: Search "Python 3.12"' + #13#10 +
              '• Python.org: Download Python 3.10 or newer' + #13#10 + #13#10 +
              'Would you like to continue with the installation anyway?', 
              mbInformation, MB_YESNO);
    
    if Response = IDNO then
    begin
      Result := False;
    end;
  end;
end;

procedure InitializeWizard();
begin
  // Custom initialization if needed
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  // Pre-installation checks
  Result := '';
  NeedsRestart := False;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ProgressPage: TOutputProgressWizardPage;
begin
  if CurStep = ssInstall then
  begin
    // Show custom progress for dependency installation
    ProgressPage := CreateOutputProgressPage('Installing Dependencies', 'Please wait while Python dependencies are being installed...');
    try
      ProgressPage.Show;
      ProgressPage.SetText('Preparing installation...', '');
      ProgressPage.SetProgress(0, 100);
      
      // Let the installer continue
    finally
      ProgressPage.Hide;
    end;
  end
  else if CurStep = ssPostInstall then
  begin
    // Create .streamlit directories if they don't exist
    if not DirExists(ExpandConstant('{app}\.streamlit')) then
      CreateDir(ExpandConstant('{app}\.streamlit'));
    if not DirExists(ExpandConstant('{app}\Forecaster App\.streamlit')) then
      CreateDir(ExpandConstant('{app}\Forecaster App\.streamlit'));
    if not DirExists(ExpandConstant('{app}\Quarter Outlook App\.streamlit')) then
      CreateDir(ExpandConstant('{app}\Quarter Outlook App\.streamlit'));
  end;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  // Skip pages based on custom logic if needed
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  // Handle page changes if needed
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  
  // Add custom validation for specific pages if needed
  if CurPageID = wpSelectComponents then
  begin
    // Warn user if they deselect automatic dependency installation
    if not IsComponentSelected('autodeps') and IsPythonInstalled() then
    begin
      if MsgBox('You have deselected automatic Python dependency installation.' + #13#10 + 
                'You will need to run SETUP.bat manually after installation.' + #13#10 + #13#10 +
                'Continue anyway?', mbConfirmation, MB_YESNO) = IDNO then
      begin
        Result := False;
      end;
    end;
  end;
end;

[Messages]
; Custom messages
WelcomeLabel2=This will install [name/ver] on your computer.%n%nThis suite includes two powerful forecasting applications:%n%n• Main Forecasting App - Advanced time series forecasting%n• Quarterly Outlook Forecaster - Revenue projections%n%nBoth applications use machine learning models including Prophet, ARIMA, LightGBM, and XGBoost for accurate predictions.%n%nThe installer can automatically set up Python dependencies for you.%n%nIt is recommended that you close all other applications before continuing.

FinishedLabel=Setup has finished installing [name] on your computer.%n%nIf you selected automatic dependency installation, Python packages have been installed and the applications are ready to use.%n%nTo get started:%n• Launch applications from the Start Menu or Desktop shortcuts%n• Refer to the documentation for usage instructions%n%nIf dependency installation was skipped, run SETUP.bat in the installation folder before using the applications.

SelectComponentsLabel2=Select the components you would like to install; clear the components you do not want to install. Click Next when you are ready to continue.%n%nNote: Automatic Python dependency installation is recommended for first-time users. This will run SETUP.bat automatically after file installation.

ComponentsDiskSpaceGBLabel=At least [gb] GB of free disk space is required.

ComponentsDiskSpaceMBLabel=At least [mb] MB of free disk space is required (including space for Python dependencies).
