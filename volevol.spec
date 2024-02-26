# -*- mode: python ; coding: utf-8 -*-

# increase recursion limit just in case...
import sys 
sys.setrecursionlimit(5000)

# need to explicitly add glfw and libsvm binaries
from PyInstaller.utils.hooks import collect_dynamic_libs
glfwBinaries = collect_dynamic_libs('glfw')

a = Analysis(
    ['volevol.py'],
    pathex=[],
    binaries = glfwBinaries,
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='volevol',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='volevol',
)
