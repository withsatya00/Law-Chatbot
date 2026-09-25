<#
.SYNOPSIS
    Shared host-environment repairs for running this project on Windows.
    Dot-source it: `. "$PSScriptRoot\_win_env.ps1"`.

.DESCRIPTION
    Changes NO application behaviour. It fixes two things about the *host* that
    would otherwise stop the stock commands in README.md from working, and is
    explained in full in WINDOWS_SETUP.md.

    1. SSLKEYLOGFILE
       Avast Antivirus injects `SSLKEYLOGFILE=\.\aswMonFltProxy\<id>` into
       every process it monitors. Python's `ssl.create_default_context()` hands
       that to OpenSSL's `BIO_new_file()`; this CPython build has no applink
       stub for a device path, so OpenSSL calls abort():

           OPENSSL_Uplink(0x...,08): no OPENSSL_Applink

       The interpreter dies with no traceback -- on `import app.main` (WeasyPrint
       builds a URLFetcher at import time) and on any outbound HTTPS call. The
       variable is not in the User or Machine registry (Avast injects it at
       process creation), so it can only be cleared per-process.

    2. PATH ORDER -- LOAD-BEARING, do not reorder.
       UB-Mannheim's Tesseract build ships its OWN copy of the GTK/Pango stack
       (libgobject-2.0-0.dll, libpango-1.0-0.dll, libharfbuzz-0.dll, ...) built
       against mingw64. C:\msys64\ucrt64\bin ships those same library NAMES built
       against ucrt64. WeasyPrint resolves each library independently, so if
       Tesseract's directory is searched first the process loads Tesseract's
       libgobject alongside MSYS2's libpango and PDF export dies with
       `error 0x7f` (ERROR_PROC_NOT_FOUND), taking `import app.drafting.export`
       -- and therefore `import app.main` -- down with it.

       MSYS2 first makes the whole GTK stack resolve from one consistent build.
       Tesseract is unaffected by being later on PATH: Windows searches an
       .exe's own directory before PATH when resolving its DLLs, so
       tesseract.exe still loads its own copies.

.NOTES
    This script DISCOVERS paths; it does not verify features. Probing a
    directory the current user cannot enumerate emits a non-terminating
    "Access is denied" record from `Test-Path`/`Get-ChildItem` even though a
    directory that cannot be read is simply one this script will not add --
    the same outcome as the directory not existing. Those probes are therefore
    silenced (`-ErrorAction SilentlyContinue`) so the launchers start clean.

    Nothing is hidden by that: a genuinely missing or broken dependency is
    reported, by name and with the install command, by

        .venv\Scripts\python.exe scripts\validate_environment.py

    which checks the Tesseract and Poppler BINARIES, the WeasyPrint native
    libraries, and every Python package -- rather than inferring health from
    whether a directory happened to be listable.
#>

Remove-Item Env:\SSLKEYLOGFILE -ErrorAction SilentlyContinue

function Resolve-PopplerBin {
    <#
        WinGet installs Poppler under a version-stamped directory
        (`poppler-25.07.0\Library\bin`). Hard-coding that version meant the
        next `winget upgrade` silently moved the directory out from under this
        script: nothing failed here, `pdf2image` just stopped finding
        pdftoppm and OCR of scanned PDFs began returning empty text with no
        error anywhere. Resolved by pattern, newest version last-wins, so an
        upgrade is picked up automatically.
    #>
    $packageRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    $candidates = Get-ChildItem -Path $packageRoot -Directory -Filter 'oschwartz10612.Poppler*' -ErrorAction SilentlyContinue
    foreach ($package in $candidates) {
        $versions = Get-ChildItem -Path $package.FullName -Directory -Filter 'poppler-*' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        foreach ($version in $versions) {
            $bin = Join-Path $version.FullName 'Library\bin'
            if (Test-Path -LiteralPath $bin -ErrorAction SilentlyContinue) { return $bin }
        }
    }
    return $null
}

# Order is load-bearing -- see the PATH ORDER note above.
$candidateDirs = @(
    'C:\msys64\ucrt64\bin'
    'C:\Program Files\Tesseract-OCR'
    (Resolve-PopplerBin)
)

$requiredDirs = @(
    $candidateDirs |
        Where-Object { $_ } |
        Where-Object { Test-Path -LiteralPath $_ -ErrorAction SilentlyContinue }
)

$rest = $env:PATH -split ';' | Where-Object { $_ -and $requiredDirs -notcontains $_ }
$env:PATH = (@($requiredDirs) + @($rest)) -join ';'
