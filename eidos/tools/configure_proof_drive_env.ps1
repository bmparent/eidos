param(
    [string]$DriveRoot,
    [switch]$PersistUser,
    [switch]$CheckOnly,
    [switch]$AllowUnverifiedLocalFolder
)

$ErrorActionPreference = "Stop"

function Convert-ToProofPath {
    param([string]$PathValue)
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $null
    }
    return [System.IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($PathValue))
}

function Test-WritableDirectory {
    param([string]$PathValue)
    if (-not (Test-Path -LiteralPath $PathValue -PathType Container)) {
        return $false
    }
    $testPath = Join-Path $PathValue (".eidos_drive_write_test_{0}.tmp" -f ([Guid]::NewGuid().ToString("N")))
    try {
        Set-Content -LiteralPath $testPath -Value "eidos proof drive validation" -Encoding UTF8
        Remove-Item -LiteralPath $testPath -Force
        return $true
    } catch {
        if (Test-Path -LiteralPath $testPath) {
            Remove-Item -LiteralPath $testPath -Force -ErrorAction SilentlyContinue
        }
        return $false
    }
}

function Test-VerifiedDriveRoot {
    param([string]$PathValue)
    if ($AllowUnverifiedLocalFolder) {
        return $true
    }

    $normalized = $PathValue.TrimEnd("\")
    if ($normalized -match "^[D-Zd-z]:\\My Drive$") {
        return $true
    }
    if ($normalized -match "^[D-Zd-z]:\\Shared drives$") {
        return $true
    }
    if ($normalized -like "*\GoogleDriveFS\*" -or $normalized -like "*\DriveFS\*") {
        return $true
    }
    return $false
}

function Get-CandidateRoots {
    $candidates = New-Object System.Collections.Generic.List[string]
    foreach ($envName in @("EIDOS_PROOF_DRIVE_DIR", "EIDOS_ARTIFACT_ROOT")) {
        $value = [Environment]::GetEnvironmentVariable($envName, "Process")
        if (-not $value) {
            $value = [Environment]::GetEnvironmentVariable($envName, "User")
        }
        if ($value) {
            $candidates.Add($value)
        }
    }

    foreach ($drive in [System.IO.DriveInfo]::GetDrives()) {
        if ($drive.DriveType -eq [System.IO.DriveType]::Fixed -or $drive.DriveType -eq [System.IO.DriveType]::Network) {
            $candidates.Add((Join-Path $drive.RootDirectory.FullName "My Drive"))
            $candidates.Add((Join-Path $drive.RootDirectory.FullName "Shared drives"))
        }
    }

    if ($DriveRoot) {
        $candidates.Insert(0, $DriveRoot)
    }
    return $candidates | Select-Object -Unique
}

$checked = New-Object System.Collections.Generic.List[object]
$selected = $null

foreach ($candidate in Get-CandidateRoots) {
    $expanded = Convert-ToProofPath $candidate
    if (-not $expanded) {
        continue
    }
    $exists = Test-Path -LiteralPath $expanded -PathType Container
    $verified = $false
    $writable = $false
    if ($exists) {
        $verified = Test-VerifiedDriveRoot $expanded
        if ($verified) {
            $writable = Test-WritableDirectory $expanded
        }
    }
    $checked.Add([ordered]@{
        path = $expanded
        exists = $exists
        verified_drive_root = $verified
        writable = $writable
    })
    if ($exists -and $verified -and $writable -and -not $selected) {
        $selected = $expanded
    }
}

if ($selected -and $PersistUser -and -not $CheckOnly) {
    [Environment]::SetEnvironmentVariable("EIDOS_PROOF_DRIVE_DIR", $selected, "User")
}

$status = [ordered]@{
    found = [bool]$selected
    selected_drive_root = $(if ($selected) { $selected } else { $null })
    persisted_user_env = [bool]($selected -and $PersistUser -and -not $CheckOnly)
    env_var = "EIDOS_PROOF_DRIVE_DIR"
    checked = $checked
    next_step = $(if ($selected) {
        "Run proof commands from a new shell, or set `$env:EIDOS_PROOF_DRIVE_DIR='$selected' in this shell."
    } else {
        "Install and sign into Google Drive Desktop so a real 'My Drive' mount appears, usually G:\My Drive, then rerun this helper with -PersistUser."
    })
}

$status | ConvertTo-Json -Depth 6

if (-not $selected -and -not $CheckOnly) {
    exit 2
}
