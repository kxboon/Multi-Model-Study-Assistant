<#
.SYNOPSIS
    Launch the moderated user-study stack: FastAPI, Streamlit, and an ngrok tunnel.

.DESCRIPTION
    Preflights Ollama and the ngrok credential, starts the two local servers,
    warms the LLM so the first participant does not pay the cold-load cost, then
    opens a single public tunnel to Streamlit (8501) and prints the link.

    FastAPI (8000) and Ollama (11434) stay bound to 127.0.0.1. Every requests.*
    call lives in frontend/app.py and runs server-side, so the participant's
    browser only ever talks to Streamlit. One tunnel is sufficient.

    SECURITY: the tunnel is unauthenticated. Anyone holding the URL has full
    ingest and query access to the vector store. Share it per participant and
    stop the tunnel between sessions.

.PARAMETER WarmAppModule
    Optional. A module name that already holds chunks (e.g. "CM3060 NLP"). When
    given, the script additionally issues one POST /query against it to load the
    embedder (~3.4s) and the sentiment model (~2s) inside the uvicorn process.
    This WRITES one record to query_debug.json and one to signals.json. Left
    empty by default so the freshly-reset study logs stay clean.
#>
[CmdletBinding()]
param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501,
    [string]$ExpectedModel = "llama3.2",
    [string]$WarmAppModule = ""
)

$ErrorActionPreference = "Stop"

function Info($m) { Write-Host "  $m" }
function Ok($m)   { Write-Host "  OK    $m" -ForegroundColor Green }
function Fail($m) { Write-Host "  FAIL  $m" -ForegroundColor Red }

# --- 0. Always run from the repo root ---------------------------------------
# DEBUG_PATH in backend/retrieve.py is a bare relative Path("query_debug.json"),
# so it resolves against the CWD of whichever process writes it. A server
# started from anywhere else silently writes its log to the wrong directory.
# Anchor to this script's own folder, which is the repo root.
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host ""
Write-Host "=== Study stack ===" -ForegroundColor Cyan
Info "repo root : $RepoRoot"

$Py = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Fail "venv not found at $Py"; exit 1 }
Info "python    : $Py"

$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Force UTF-8 for the child processes' stdout/stderr. Redirecting a Python
# process's output to a file makes it fall back to the locale encoding (cp1252
# here) instead of the console's Unicode path, so the [QUIZ RAW] / [FLASHCARD
# RAW] debug prints in backend/main.py raise UnicodeEncodeError the moment the
# model emits a character outside cp1252 - a Greek letter, an arrow, an emoji.
# That exception escapes the endpoint and the participant sees a 500. Start-Process
# inherits this environment, so setting it here covers uvicorn and streamlit both.
$env:PYTHONUTF8 = "1"

# --- 1. Ollama up? -----------------------------------------------------------
Write-Host ""
Write-Host "[1/6] Ollama" -ForegroundColor Cyan
try {
    $ver = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 5
    Ok "ollama responding (version $($ver.version))"
} catch {
    Fail "no response on 127.0.0.1:11434 - start it with: ollama serve"
    exit 1
}

# --- 2. OLLAMA_MODEL resolves to the expected tag? ---------------------------
Write-Host ""
Write-Host "[2/6] Model" -ForegroundColor Cyan
$EnvFile = Join-Path $RepoRoot ".env"
$Model = $null
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match '^\s*OLLAMA_MODEL\s*=\s*([^#\s]+)') { $Model = $Matches[1] }
    }
}
if (-not $Model) {
    # backend/retrieve.py falls back to "llama3" when the var is unset.
    $Model = "llama3"
    Info "OLLAMA_MODEL not set in .env - backend default applies"
}
Info ".env OLLAMA_MODEL = $Model"

if ($Model -ne $ExpectedModel) {
    Fail "expected '$ExpectedModel' but .env resolves to '$Model'"
    exit 1
}

$tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 15
$pulled = @($tags.models | ForEach-Object { $_.name })
$match = $pulled | Where-Object { $_ -eq $Model -or $_ -eq ($Model + ":latest") }
if (-not $match) {
    Fail "'$Model' is not pulled. Available: $($pulled -join ', ')"
    Info "pull it with: ollama pull $Model"
    exit 1
}
Ok "$Model resolves and is pulled (as '$match')"

# --- 3. ngrok credential present? -------------------------------------------
# ngrok v3 refuses to open any tunnel, free tier included, without an authtoken
# (ERR_NGROK_4018). Check before starting servers so a failure costs nothing.
Write-Host ""
Write-Host "[3/6] ngrok credential" -ForegroundColor Cyan
$ngrokCmd = Get-Command ngrok -ErrorAction SilentlyContinue
if (-not $ngrokCmd) { Fail "ngrok is not on PATH"; exit 1 }
Info "ngrok     : $($ngrokCmd.Source)"

# PowerShell 5.1 wraps a native command's redirected stderr in a NativeCommandError,
# which $ErrorActionPreference = "Stop" would turn into a terminating error before
# the clean message below ever prints. Relax it just for this call.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$cfgCheck = & ngrok config check 2>&1
$cfgExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP

if ($cfgExit -ne 0) {
    Fail "no usable ngrok config / authtoken"
    # .ToString() on the ErrorRecord keeps ngrok's own message and drops
    # PowerShell's "At <script>:<line> char:" stack trace around it.
    foreach ($l in $cfgCheck) { Info ($l.ToString().Trim()) }
    Info "fix with: ngrok config add-authtoken <token from dashboard.ngrok.com>"
    exit 1
}
Ok "ngrok config valid"

# --- 4. Start the local servers ---------------------------------------------
Write-Host ""
Write-Host "[4/6] Local servers" -ForegroundColor Cyan

$api = Start-Process -FilePath $Py -PassThru -WorkingDirectory $RepoRoot `
    -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "$ApiPort") `
    -RedirectStandardOutput (Join-Path $LogDir "api.out.log") `
    -RedirectStandardError (Join-Path $LogDir "api.err.log")
Info "uvicorn   pid $($api.Id) -> logs\api.*.log"

# --server.headless stops Streamlit opening a local browser and suppresses its
# first-run email prompt, which would otherwise block startup.
$ui = Start-Process -FilePath $Py -PassThru -WorkingDirectory $RepoRoot `
    -ArgumentList @("-m", "streamlit", "run", "frontend/app.py",
                    "--server.port", "$UiPort", "--server.headless", "true") `
    -RedirectStandardOutput (Join-Path $LogDir "ui.out.log") `
    -RedirectStandardError (Join-Path $LogDir "ui.err.log")
Info "streamlit pid $($ui.Id) -> logs\ui.*.log"

# Wait for FastAPI to answer /health.
$h = $null
foreach ($i in 1..40) {
    Start-Sleep -Milliseconds 750
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 3
        if ($h.status -eq "ok") { break }
    } catch { $h = $null }
}
if (-not $h) { Fail "FastAPI did not come up - see logs\api.err.log"; exit 1 }
Ok "FastAPI healthy on 127.0.0.1:$ApiPort (ollama=$($h.ollama))"

# Wait for Streamlit to accept connections.
$uiReady = $false
foreach ($i in 1..40) {
    Start-Sleep -Milliseconds 750
    $sock = New-Object Net.Sockets.TcpClient
    try {
        $sock.Connect("127.0.0.1", $UiPort)
        if ($sock.Connected) { $uiReady = $true }
    } catch { }
    finally { $sock.Close() }
    if ($uiReady) { break }
}
if (-not $uiReady) { Fail "Streamlit did not come up - see logs\ui.err.log"; exit 1 }
Ok "Streamlit listening on $UiPort"

# --- 5. Warm the model ------------------------------------------------------
# Straight to Ollama, not through /query: POST /query returns 404 before it ever
# reaches Ollama when the module holds no chunks, and any /query that does reach
# it writes a record to query_debug.json and signals.json. This warms the LLM
# without touching the study logs.
Write-Host ""
Write-Host "[5/6] Warm-up" -ForegroundColor Cyan
$warmBody = @{ model = $Model; prompt = "Reply with the single word: ready."; stream = $false } | ConvertTo-Json
$sw = [Diagnostics.Stopwatch]::StartNew()
$warm = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" `
    -Method Post -Body $warmBody -ContentType "application/json" -TimeoutSec 600
$sw.Stop()
if ($warm.eval_duration -gt 0) {
    $genRate = [math]::Round($warm.eval_count / ($warm.eval_duration / 1e9), 1)
} else {
    $genRate = 0
}
Ok ("$Model warm - {0} tokens, {1} tok/s gen, {2:N1}s wall (load {3:N1}s)" -f `
    $warm.eval_count, $genRate, $sw.Elapsed.TotalSeconds, ($warm.load_duration / 1e9))

if ($WarmAppModule -ne "") {
    Info "app warm-up against '$WarmAppModule' (writes 1 query_debug + 1 signals record)"
    $qBody = @{ question = "warm-up"; session_id = $WarmAppModule } | ConvertTo-Json
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/query" -Method Post `
            -Body $qBody -ContentType "application/json" -TimeoutSec 600 | Out-Null
        Ok "embedder + sentiment loaded in the uvicorn process"
    } catch {
        Info "app warm-up failed (non-fatal): $($_.Exception.Message)"
    }
}

# --- 6. Tunnel --------------------------------------------------------------
Write-Host ""
Write-Host "[6/6] Tunnel" -ForegroundColor Cyan
$ng = Start-Process -FilePath $ngrokCmd.Source -PassThru `
    -ArgumentList @("http", "$UiPort", "--log", "stdout") `
    -RedirectStandardOutput (Join-Path $LogDir "ngrok.out.log") `
    -RedirectStandardError (Join-Path $LogDir "ngrok.err.log")
Info "ngrok     pid $($ng.Id) -> logs\ngrok.*.log"

$publicUrl = $null
foreach ($i in 1..40) {
    Start-Sleep -Milliseconds 750
    try {
        $tun = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 3
        $https = $tun.tunnels | Where-Object { $_.public_url -like "https://*" } | Select-Object -First 1
        if ($https) { $publicUrl = $https.public_url; break }
    } catch { }
}
if (-not $publicUrl) { Fail "no tunnel URL - see logs\ngrok.err.log"; exit 1 }
Ok "tunnel open"

Write-Host ""
Write-Host "=== READY ===" -ForegroundColor Green
Write-Host "  public URL       : $publicUrl"
Write-Host "  participant link : $publicUrl/?p=P01   (change P01 per participant)"
Write-Host "  ngrok inspector  : http://127.0.0.1:4040"
Write-Host ""
Write-Host "  The tunnel is UNAUTHENTICATED - anyone with the URL has full" -ForegroundColor Yellow
Write-Host "  ingest and query access. Stop it between sessions." -ForegroundColor Yellow
Write-Host ""
Write-Host "  stop everything:"
Write-Host "    Stop-Process -Id $($ng.Id),$($ui.Id),$($api.Id)"
Write-Host ""
