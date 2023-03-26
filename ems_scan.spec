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
             excludes=['matplotlib'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
a.datas += [('tw.png','C:\\Users\\pst\\tresorit\\Tofwerk\\Python\\EMS\\tw.png', 'Data'),
('tw.ico','C:\\Users\\pst\\tresorit\\Tofwerk\\Python\\EMS\\tw.ico', 'Data')]
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='EMS scan',
          icon='C:\\Users\\pst\\tresorit\\Tofwerk\\Python\\EMS\\tw.ico',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          version='versionfile.txt' )
