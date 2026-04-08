# =============================================================================
# Start All Services - COS40007 ML Pipeline
# =============================================================================
# Usage: Right-click -> "Run with PowerShell" or execute from terminal:
#   .\scripts\start_services.ps1
# =============================================================================

param(
    [switch]$AirflowOnly,
    [switch]$BentoOnly,
    [switch]$StreamlitOnly,
    [switch]$MLflowOnly
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$BackendDir  = Join-Path $ProjectRoot "backend"
$AirflowDir  = Join-Path $ProjectRoot "airflow"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  COS40007 ML Pipeline - Service Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# --- Helper ---
function Start-ServiceInNewWindow {
    param([string]$Title, [string]$Command, [string]$WorkingDir)
    Write-Host "[START] $Title" -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$WorkingDir'; $Command" -WorkingDirectory $WorkingDir
}

$startAll = -not ($AirflowOnly -or $BentoOnly -or $StreamlitOnly -or $MLflowOnly)

# 1. Apache Airflow (Docker)
if ($startAll -or $AirflowOnly) {
    Write-Host "[1/4] Starting Apache Airflow (Docker Compose)..." -ForegroundColor Yellow
    if (Test-Path $AirflowDir) {
        Start-ServiceInNewWindow -Title "Airflow" -Command "docker compose up -d; Write-Host 'Airflow UI: http://localhost:8080 (admin/admin)' -ForegroundColor Green" -WorkingDir $AirflowDir
    } else {
        Write-Host "  [!] Airflow directory not found at $AirflowDir" -ForegroundColor Red
    }
}

# 2. BentoML Serving
if ($startAll -or $BentoOnly) {
    Write-Host "[2/4] Starting BentoML Serving..." -ForegroundColor Yellow
    Start-ServiceInNewWindow -Title "BentoML" -Command "conda activate AIE; bentoml serve serving.service:MotionClassifier --reload --port 3000" -WorkingDir $BackendDir
}

# 3. MLflow UI
if ($startAll -or $MLflowOnly) {
    Write-Host "[3/4] Starting MLflow UI..." -ForegroundColor Yellow
    Start-ServiceInNewWindow -Title "MLflow" -Command "conda activate AIE; mlflow ui --backend-store-uri sqlite:///mlflow_tracking.db --port 5000" -WorkingDir $BackendDir
}

# 4. Streamlit
if ($startAll -or $StreamlitOnly) {
    Write-Host "[4/4] Starting Streamlit Dashboard..." -ForegroundColor Yellow
    Start-ServiceInNewWindow -Title "Streamlit" -Command "conda activate AIE; streamlit run ui_app.py" -WorkingDir $BackendDir
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All services launching!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Airflow   -> http://localhost:8080  (admin/admin)" -ForegroundColor White
Write-Host "  BentoML   -> http://localhost:3000" -ForegroundColor White
Write-Host "  MLflow    -> http://localhost:5000" -ForegroundColor White
Write-Host "  Streamlit -> http://localhost:8501" -ForegroundColor White
Write-Host ""
