# ForgeAI Chocolatey Install Script
$ErrorActionPreference = 'Stop'

$packageName = 'forgeai'
$toolsDir = "$(Split-Path -parent $MyInvocation.MyCommand.Definition)"
$installDir = Join-Path $env:ProgramFiles 'ForgeAI'

# Download and install
$url = 'https://github.com/forgeai/forge_ai/releases/download/v1.0.0/forgeai-windows-x64.zip'
$checksum = 'PLACEHOLDER_CHECKSUM'
$checksumType = 'sha256'

$packageArgs = @{
  packageName   = $packageName
  unzipLocation = $installDir
  url           = $url
  checksum      = $checksum
  checksumType  = $checksumType
}

Install-ChocolateyZipPackage @packageArgs

# Create virtual environment
$pythonPath = Join-Path $env:ProgramFiles 'Python311\python.exe'
if (-not (Test-Path $pythonPath)) {
    $pythonPath = 'python'
}

$venvPath = Join-Path $installDir 'venv'
& $pythonPath -m venv $venvPath

# Install dependencies
$pipPath = Join-Path $venvPath 'Scripts\pip.exe'
& $pipPath install -r (Join-Path $installDir 'requirements.txt')
& $pipPath install -e $installDir

# Add to PATH
$binPath = Join-Path $installDir 'bin'
Install-ChocolateyPath -PathToInstall $binPath -PathType 'Machine'

# Create shortcuts
$desktopPath = [Environment]::GetFolderPath('Desktop')
$startMenuPath = Join-Path $env:ProgramData 'Microsoft\Windows\Start Menu\Programs\ForgeAI'

if (-not (Test-Path $startMenuPath)) {
    New-Item -ItemType Directory -Path $startMenuPath | Out-Null
}

# GUI shortcut
$pythonVenv = Join-Path $venvPath 'Scripts\python.exe'
$shortcutPath = Join-Path $startMenuPath 'ForgeAI.lnk'
$targetPath = $pythonVenv
$arguments = '-m forge_ai.run --gui'
$iconPath = Join-Path $installDir 'assets\icon.ico'

Install-ChocolateyShortcut `
    -ShortcutFilePath $shortcutPath `
    -TargetPath $targetPath `
    -Arguments $arguments `
    -IconLocation $iconPath `
    -WorkingDirectory $installDir `
    -Description 'ForgeAI - Local AI Framework'

# Desktop shortcut
$desktopShortcut = Join-Path $desktopPath 'ForgeAI.lnk'
Install-ChocolateyShortcut `
    -ShortcutFilePath $desktopShortcut `
    -TargetPath $targetPath `
    -Arguments $arguments `
    -IconLocation $iconPath `
    -WorkingDirectory $installDir `
    -Description 'ForgeAI - Local AI Framework'

# Create batch files in bin
$binDir = Join-Path $installDir 'bin'
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}

# forgeai.bat
@"
@echo off
"$pythonVenv" -m forge_ai.run %*
"@ | Out-File -FilePath (Join-Path $binDir 'forgeai.bat') -Encoding ASCII

# forgeai-gui.bat  
@"
@echo off
start "" "$pythonVenv" -m forge_ai.run --gui %*
"@ | Out-File -FilePath (Join-Path $binDir 'forgeai-gui.bat') -Encoding ASCII

Write-Host "ForgeAI has been installed successfully!"
Write-Host "Run 'forgeai' or 'forgeai-gui' from the command line."
