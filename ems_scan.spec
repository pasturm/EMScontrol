# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['ems_scan.pyw'],
             pathex=['C:\\Users\\pst\\tresorit\\Tofwerk\\Python\\EMS'],
             binaries=[('C:\\Users\\pst\\tresorit\\Tofwerk\\Python\\EMS\\TofDaqDll.dll', '.'),
             ('C:\\Users\\pst\\tresorit\\Tofwerk\\Python\\EMS\\TwToolDll.dll', '.')],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='EMS scan | TOFWERK',
          icon='C:\\Users\\pst\\tresorit\\Tofwerk\\Python\\EMS\\tw.ico',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False )
