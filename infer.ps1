$body = Get-Content -Raw "C:\Users\hasan\.clone\vuln-triage-harness\test_request.json"
$result = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/serve" -Method Post -Headers @{"Content-Type"="application/json"} -Body $body
Write-Output $result | ConvertTo-Json -Depth 10
