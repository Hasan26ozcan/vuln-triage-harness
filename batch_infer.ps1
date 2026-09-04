$body = Get-Content -Raw "C:\Users\hasan\.clone\vuln-triage-harness\batch_test.json"
$result = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/serve/batch" -Method Post -Headers @{"Content-Type"="application/json"} -Body $body
Write-Output "=== BATCH RESULTS ==="
$result | ConvertTo-Json -Depth 10
Write-Output ""
Write-Output "=== MANIFEST ==="
$manifest = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/manifest" -Method Get
$manifest | ConvertTo-Json -Depth 10
