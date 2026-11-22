<#
  Windows equivalent of the Makefile.  Usage:  .\scripts\dev.ps1 <target>
  Targets: install, pin, lint, fmt, types, test, check, run, qdrant, ready
#>
param([Parameter(Mandatory = $true)][string]$Target)

$ErrorActionPreference = 'Stop'
$py = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'

function Invoke-Step([scriptblock]$Block) {
    & $Block
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    'install' {
        Invoke-Step { uv venv --python 3.12 .venv }
        Invoke-Step { uv pip install --python .venv -r requirements.txt -r requirements-dev.txt }
    }
    'pin' {
        Invoke-Step { uv pip compile --universal requirements.in -o requirements.txt }
        Invoke-Step { uv pip compile --universal requirements-dev.in -o requirements-dev.txt }
    }
    'lint' {
        Invoke-Step { & $py -m ruff check app tests }
        Invoke-Step { & $py -m ruff format --check app tests }
    }
    'fmt' {
        Invoke-Step { & $py -m ruff check app tests --fix }
        Invoke-Step { & $py -m ruff format app tests }
    }
    'types' { Invoke-Step { & $py -m mypy } }
    'test' { Invoke-Step { & $py -m pytest -m 'not live' } }
    'check' {
        Invoke-Step { & $py -m ruff check app tests }
        Invoke-Step { & $py -m ruff format --check app tests }
        Invoke-Step { & $py -m mypy }
        Invoke-Step { & $py -m pytest -m 'not live' }
    }
    'run' { Invoke-Step { & $py -m uvicorn app.main:app_factory --factory --reload --port 8000 } }
    'qdrant' { Invoke-Step { docker compose up -d qdrant } }
    'ready' { Invoke-RestMethod http://127.0.0.1:8000/health/ready -SkipHttpErrorCheck | ConvertTo-Json -Depth 5 }
    default { Write-Error "Unknown target '$Target'"; exit 2 }
}
