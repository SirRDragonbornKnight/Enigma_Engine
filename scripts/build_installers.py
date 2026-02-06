"""
Platform Installer Builder

Build platform-specific installers for ForgeAI.
Supports MSI/EXE (Windows), DMG (macOS), AppImage/Flatpak (Linux).

FILE: scripts/build_installers.py
TYPE: Build/Packaging
MAIN CLASSES: InstallerBuilder, WindowsBuilder, MacOSBuilder, LinuxBuilder
"""

import logging
import subprocess
import shutil
import json
import os
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported platforms."""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


@dataclass
class BuildConfig:
    """Configuration for installer builds."""
    app_name: str = "ForgeAI"
    version: str = "1.0.0"
    description: str = "AI Framework with Modular Design"
    author: str = "ForgeAI Team"
    license: str = "MIT"
    
    # Paths
    source_dir: Path = None
    build_dir: Path = None
    output_dir: Path = None
    
    # Python
    python_version: str = "3.10"
    
    # Icons
    icon_ico: str = ""  # Windows icon
    icon_icns: str = ""  # macOS icon
    icon_png: str = ""  # Linux icon
    
    # Signing
    sign_windows: bool = False
    sign_macos: bool = False
    certificate_path: str = ""
    
    def __post_init__(self):
        if self.source_dir is None:
            self.source_dir = Path(__file__).parent.parent
        if self.build_dir is None:
            self.build_dir = self.source_dir / "build"
        if self.output_dir is None:
            self.output_dir = self.source_dir / "dist"


class InstallerBuilder:
    """Base class for platform-specific installer builders."""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        self.config.build_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build(self) -> Path:
        """Build the installer. Override in subclasses."""
        raise NotImplementedError
    
    def _run_command(self, cmd: List[str], cwd: Path = None) -> bool:
        """Run command and check result."""
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.config.source_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Command failed: {result.stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False
    
    def _create_pyinstaller_spec(self) -> Path:
        """Create PyInstaller spec file."""
        spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=['{self.config.source_dir}'],
    binaries=[],
    datas=[
        ('forge_ai', 'forge_ai'),
        ('data', 'data'),
        ('models', 'models'),
    ],
    hiddenimports=[
        'torch',
        'numpy',
        'PyQt5',
        'flask',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{self.config.app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{self.config.icon_ico}' if '{self.config.icon_ico}' else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{self.config.app_name}',
)
'''
        
        spec_path = self.config.build_dir / f"{self.config.app_name}.spec"
        spec_path.write_text(spec_content)
        return spec_path


class WindowsBuilder(InstallerBuilder):
    """Build Windows installers (MSI, EXE)."""
    
    def build(self, installer_type: str = "exe") -> Path:
        """
        Build Windows installer.
        
        Args:
            installer_type: "exe" (NSIS) or "msi" (WiX)
        """
        # First, create PyInstaller bundle
        if not self._build_pyinstaller():
            raise RuntimeError("PyInstaller build failed")
        
        if installer_type == "exe":
            return self._build_nsis()
        elif installer_type == "msi":
            return self._build_msi()
        else:
            raise ValueError(f"Unknown installer type: {installer_type}")
    
    def _build_pyinstaller(self) -> bool:
        """Build with PyInstaller."""
        spec_path = self._create_pyinstaller_spec()
        
        return self._run_command([
            "pyinstaller",
            "--noconfirm",
            "--workpath", str(self.config.build_dir / "pyinstaller"),
            "--distpath", str(self.config.build_dir / "dist"),
            str(spec_path)
        ])
    
    def _build_nsis(self) -> Path:
        """Build NSIS installer."""
        nsi_content = f'''
!include "MUI2.nsh"

Name "{self.config.app_name}"
OutFile "{self.config.output_dir}\\{self.config.app_name}-{self.config.version}-Setup.exe"
InstallDir "$PROGRAMFILES\\{self.config.app_name}"
RequestExecutionLevel admin

!define MUI_ABORTWARNING
!define MUI_ICON "{self.config.icon_ico}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath "$INSTDIR"
    File /r "{self.config.build_dir}\\dist\\{self.config.app_name}\\*"
    
    CreateDirectory "$SMPROGRAMS\\{self.config.app_name}"
    CreateShortCut "$SMPROGRAMS\\{self.config.app_name}\\{self.config.app_name}.lnk" "$INSTDIR\\{self.config.app_name}.exe"
    CreateShortCut "$DESKTOP\\{self.config.app_name}.lnk" "$INSTDIR\\{self.config.app_name}.exe"
    
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{self.config.app_name}" "DisplayName" "{self.config.app_name}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{self.config.app_name}" "UninstallString" "$INSTDIR\\Uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{self.config.app_name}" "DisplayVersion" "{self.config.version}"
SectionEnd

Section "Uninstall"
    RMDir /r "$INSTDIR"
    RMDir /r "$SMPROGRAMS\\{self.config.app_name}"
    Delete "$DESKTOP\\{self.config.app_name}.lnk"
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{self.config.app_name}"
SectionEnd
'''
        
        nsi_path = self.config.build_dir / "installer.nsi"
        nsi_path.write_text(nsi_content)
        
        # Run NSIS
        if not self._run_command(["makensis", str(nsi_path)]):
            raise RuntimeError("NSIS build failed")
        
        output_path = self.config.output_dir / f"{self.config.app_name}-{self.config.version}-Setup.exe"
        
        # Sign if configured
        if self.config.sign_windows and self.config.certificate_path:
            self._sign_windows(output_path)
        
        logger.info(f"Built Windows installer: {output_path}")
        return output_path
    
    def _build_msi(self) -> Path:
        """Build MSI using WiX."""
        wxs_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
    <Product Id="*"
             Name="{self.config.app_name}"
             Language="1033"
             Version="{self.config.version}"
             Manufacturer="{self.config.author}"
             UpgradeCode="12345678-1234-1234-1234-123456789012">
        
        <Package InstallerVersion="200" Compressed="yes" InstallScope="perMachine"/>
        
        <MajorUpgrade DowngradeErrorMessage="A newer version is already installed."/>
        <MediaTemplate EmbedCab="yes"/>
        
        <Feature Id="ProductFeature" Title="{self.config.app_name}" Level="1">
            <ComponentGroupRef Id="ProductComponents"/>
        </Feature>
        
        <Directory Id="TARGETDIR" Name="SourceDir">
            <Directory Id="ProgramFilesFolder">
                <Directory Id="INSTALLFOLDER" Name="{self.config.app_name}"/>
            </Directory>
        </Directory>
        
        <ComponentGroup Id="ProductComponents" Directory="INSTALLFOLDER">
            <!-- Components added by heat.exe -->
        </ComponentGroup>
    </Product>
</Wix>
'''
        
        wxs_path = self.config.build_dir / "installer.wxs"
        wxs_path.write_text(wxs_content)
        
        # This is simplified - real WiX build requires heat.exe to harvest files
        logger.warning("MSI build requires WiX Toolset to be installed")
        
        return self.config.output_dir / f"{self.config.app_name}-{self.config.version}.msi"
    
    def _sign_windows(self, path: Path):
        """Sign Windows executable."""
        self._run_command([
            "signtool", "sign",
            "/f", self.config.certificate_path,
            "/t", "http://timestamp.digicert.com",
            str(path)
        ])


class MacOSBuilder(InstallerBuilder):
    """Build macOS installers (DMG, PKG)."""
    
    def build(self, installer_type: str = "dmg") -> Path:
        """Build macOS installer."""
        # Build app bundle
        if not self._build_app_bundle():
            raise RuntimeError("App bundle build failed")
        
        if installer_type == "dmg":
            return self._build_dmg()
        elif installer_type == "pkg":
            return self._build_pkg()
        else:
            raise ValueError(f"Unknown installer type: {installer_type}")
    
    def _build_app_bundle(self) -> bool:
        """Build .app bundle with py2app or PyInstaller."""
        # Use PyInstaller for macOS
        spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=['{self.config.source_dir}'],
    datas=[('forge_ai', 'forge_ai'), ('data', 'data')],
    hiddenimports=['torch', 'numpy', 'PyQt5'],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{self.config.app_name}',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='{self.config.app_name}',
)

app = BUNDLE(
    coll,
    name='{self.config.app_name}.app',
    icon='{self.config.icon_icns}' if '{self.config.icon_icns}' else None,
    bundle_identifier='com.forgeai.app',
    info_plist={{
        'CFBundleShortVersionString': '{self.config.version}',
        'CFBundleVersion': '{self.config.version}',
        'NSHighResolutionCapable': True,
    }},
)
'''
        
        spec_path = self.config.build_dir / f"{self.config.app_name}_macos.spec"
        spec_path.write_text(spec_content)
        
        return self._run_command([
            "pyinstaller",
            "--noconfirm",
            "--workpath", str(self.config.build_dir / "pyinstaller"),
            "--distpath", str(self.config.build_dir / "dist"),
            str(spec_path)
        ])
    
    def _build_dmg(self) -> Path:
        """Build DMG disk image."""
        app_path = self.config.build_dir / "dist" / f"{self.config.app_name}.app"
        dmg_path = self.config.output_dir / f"{self.config.app_name}-{self.config.version}.dmg"
        
        # Create DMG
        self._run_command([
            "hdiutil", "create",
            "-volname", self.config.app_name,
            "-srcfolder", str(app_path),
            "-ov",
            "-format", "UDZO",
            str(dmg_path)
        ])
        
        # Sign if configured
        if self.config.sign_macos:
            self._sign_macos(dmg_path)
        
        logger.info(f"Built macOS DMG: {dmg_path}")
        return dmg_path
    
    def _build_pkg(self) -> Path:
        """Build PKG installer."""
        app_path = self.config.build_dir / "dist" / f"{self.config.app_name}.app"
        pkg_path = self.config.output_dir / f"{self.config.app_name}-{self.config.version}.pkg"
        
        self._run_command([
            "pkgbuild",
            "--root", str(app_path.parent),
            "--identifier", "com.forgeai.app",
            "--version", self.config.version,
            "--install-location", "/Applications",
            str(pkg_path)
        ])
        
        logger.info(f"Built macOS PKG: {pkg_path}")
        return pkg_path
    
    def _sign_macos(self, path: Path):
        """Sign macOS app/dmg."""
        self._run_command([
            "codesign",
            "--deep",
            "--force",
            "--verify",
            "--verbose",
            "--sign", "Developer ID Application",
            str(path)
        ])


class LinuxBuilder(InstallerBuilder):
    """Build Linux packages (AppImage, Flatpak, Deb, RPM)."""
    
    def build(self, package_type: str = "appimage") -> Path:
        """Build Linux package."""
        if package_type == "appimage":
            return self._build_appimage()
        elif package_type == "flatpak":
            return self._build_flatpak()
        elif package_type == "deb":
            return self._build_deb()
        elif package_type == "rpm":
            return self._build_rpm()
        else:
            raise ValueError(f"Unknown package type: {package_type}")
    
    def _build_appimage(self) -> Path:
        """Build AppImage."""
        # Create AppDir structure
        appdir = self.config.build_dir / "AppDir"
        appdir.mkdir(parents=True, exist_ok=True)
        
        # Create desktop file
        desktop_content = f'''[Desktop Entry]
Type=Application
Name={self.config.app_name}
Exec=AppRun
Icon={self.config.app_name.lower()}
Categories=Development;
Comment={self.config.description}
'''
        
        (appdir / f"{self.config.app_name.lower()}.desktop").write_text(desktop_content)
        
        # Create AppRun
        apprun_content = '''#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/forgeai" "$@"
'''
        
        apprun_path = appdir / "AppRun"
        apprun_path.write_text(apprun_content)
        apprun_path.chmod(0o755)
        
        # Copy application
        usr_bin = appdir / "usr" / "bin"
        usr_bin.mkdir(parents=True, exist_ok=True)
        
        # Run PyInstaller
        self._run_command([
            "pyinstaller",
            "--onedir",
            "--name", "forgeai",
            "--distpath", str(usr_bin),
            "run.py"
        ])
        
        # Copy icon
        if self.config.icon_png:
            shutil.copy(self.config.icon_png, appdir / f"{self.config.app_name.lower()}.png")
        
        # Build AppImage
        output = self.config.output_dir / f"{self.config.app_name}-{self.config.version}-x86_64.AppImage"
        
        self._run_command([
            "appimagetool",
            str(appdir),
            str(output)
        ])
        
        logger.info(f"Built AppImage: {output}")
        return output
    
    def _build_flatpak(self) -> Path:
        """Build Flatpak package."""
        manifest = {
            "app-id": "com.forgeai.app",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "22.08",
            "sdk": "org.freedesktop.Sdk",
            "command": "forgeai",
            "finish-args": [
                "--share=network",
                "--socket=x11",
                "--socket=wayland",
                "--device=dri",
                "--filesystem=home"
            ],
            "modules": [
                {
                    "name": "forgeai",
                    "buildsystem": "simple",
                    "build-commands": [
                        "pip3 install --prefix=/app ."
                    ],
                    "sources": [
                        {
                            "type": "dir",
                            "path": str(self.config.source_dir)
                        }
                    ]
                }
            ]
        }
        
        manifest_path = self.config.build_dir / "com.forgeai.app.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        # Build
        self._run_command([
            "flatpak-builder",
            "--force-clean",
            str(self.config.build_dir / "flatpak-build"),
            str(manifest_path)
        ])
        
        output = self.config.output_dir / f"{self.config.app_name}-{self.config.version}.flatpak"
        logger.info(f"Built Flatpak: {output}")
        return output
    
    def _build_deb(self) -> Path:
        """Build Debian package."""
        # Create package structure
        pkg_dir = self.config.build_dir / "deb"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        
        # DEBIAN control
        debian_dir = pkg_dir / "DEBIAN"
        debian_dir.mkdir(exist_ok=True)
        
        control_content = f'''Package: {self.config.app_name.lower()}
Version: {self.config.version}
Section: devel
Priority: optional
Architecture: amd64
Maintainer: {self.config.author}
Description: {self.config.description}
'''
        
        (debian_dir / "control").write_text(control_content)
        
        # Install files
        opt_dir = pkg_dir / "opt" / self.config.app_name.lower()
        opt_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy application
        shutil.copytree(
            self.config.source_dir / "forge_ai",
            opt_dir / "forge_ai"
        )
        
        # Build
        output = self.config.output_dir / f"{self.config.app_name.lower()}_{self.config.version}_amd64.deb"
        
        self._run_command(["dpkg-deb", "--build", str(pkg_dir), str(output)])
        
        logger.info(f"Built DEB: {output}")
        return output
    
    def _build_rpm(self) -> Path:
        """Build RPM package."""
        # Create spec file
        spec_content = f'''Name: {self.config.app_name.lower()}
Version: {self.config.version}
Release: 1
Summary: {self.config.description}
License: {self.config.license}

%description
{self.config.description}

%install
mkdir -p %{{buildroot}}/opt/{self.config.app_name.lower()}
cp -r * %{{buildroot}}/opt/{self.config.app_name.lower()}/

%files
/opt/{self.config.app_name.lower()}
'''
        
        spec_path = self.config.build_dir / f"{self.config.app_name.lower()}.spec"
        spec_path.write_text(spec_content)
        
        # Build would use rpmbuild
        output = self.config.output_dir / f"{self.config.app_name.lower()}-{self.config.version}.x86_64.rpm"
        logger.warning("RPM build requires rpmbuild to be installed")
        
        return output


def detect_platform() -> Platform:
    """Detect current platform."""
    system = platform.system().lower()
    if system == "windows":
        return Platform.WINDOWS
    elif system == "darwin":
        return Platform.MACOS
    elif system == "linux":
        return Platform.LINUX
    raise RuntimeError(f"Unsupported platform: {system}")


def build_installer(
    config: BuildConfig = None,
    platform_override: Platform = None,
    installer_type: str = None
) -> Path:
    """
    Build installer for current or specified platform.
    
    Args:
        config: Build configuration
        platform_override: Override detected platform
        installer_type: Type of installer to build
    
    Returns:
        Path to built installer
    """
    config = config or BuildConfig()
    target_platform = platform_override or detect_platform()
    
    if target_platform == Platform.WINDOWS:
        builder = WindowsBuilder(config)
        return builder.build(installer_type or "exe")
    
    elif target_platform == Platform.MACOS:
        builder = MacOSBuilder(config)
        return builder.build(installer_type or "dmg")
    
    elif target_platform == Platform.LINUX:
        builder = LinuxBuilder(config)
        return builder.build(installer_type or "appimage")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ForgeAI installers")
    parser.add_argument("--platform", choices=["windows", "macos", "linux"],
                        help="Target platform")
    parser.add_argument("--type", help="Installer type (exe, msi, dmg, pkg, appimage, deb, rpm)")
    parser.add_argument("--version", default="1.0.0", help="Version number")
    parser.add_argument("--output", type=Path, help="Output directory")
    
    args = parser.parse_args()
    
    config = BuildConfig(version=args.version)
    if args.output:
        config.output_dir = args.output
    
    platform_map = {
        "windows": Platform.WINDOWS,
        "macos": Platform.MACOS,
        "linux": Platform.LINUX
    }
    
    platform_override = platform_map.get(args.platform) if args.platform else None
    
    try:
        output = build_installer(config, platform_override, args.type)
        print(f"Built: {output}")
    except Exception as e:
        logger.error(f"Build failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
