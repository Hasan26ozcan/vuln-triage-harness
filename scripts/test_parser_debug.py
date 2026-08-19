"""Debug the parser with a simulated problematic response."""

import time

from app.evaluation.parser import _extract_json, _find_json_objects, parse_prediction

# Simulated response that looks like what the model produces for sample 2
# The response has TWO ```json blocks - the second one is incomplete
raw = "```json\n"
raw += "{\n"
raw += '  "cwe_id": "CWE-79",\n'
raw += '  "severity": "medium",\n'
raw += '  "explanation": "Vulnerability patched in CVE CVE-2020-26853.",\n'
raw += (
    '  "patch_diff": "--- a/vulnerable_code\n'
    "+++ b/fixed_code\n"
    "@@ -2,7 +2,7 @@\n"
    " app.get('/search', (req, res) => {\n"
    "     const q = req.query.q;\n"
    " \n"
    " -    res.send('<h1>Results for ' + q + '</h1>');\n"
    "+    res.send(`<h1>Results for ${q}</h1>`);\n"
    " });\n"
    '",\n'
)
raw += (
    '  "patch_diff": "--- a/vulnerable_code\n'
    "+++ b/fixed_code\n"
    "@@ -2,7 +2,7 @@\n"
    " app.get('/search', (req, res) => {\n"
    "     const q = req.query.q;\n"
    " \n"
    " -    res.send('<h1>Results for ' + q + '</h1>');\n"
    "+    res.send(`<h1>Results for ${q}</h1>`);\n"
    " });\n"
    "```json"
)

print(f"Raw length: {len(raw)} chars")
print(f"Raw first 500: {repr(raw[:500])}")
print(f"Raw last 100: {repr(raw[-100:])}")
print()

t0 = time.time()

# Step 1: Try _extract_json
print("Step 1: _extract_json...", flush=True)
t1 = time.time()
json_str = _extract_json(raw)
t2 = time.time()
print(f"  _extract_json: {t2 - t1:.3f}s", flush=True)
if json_str:
    print(f"  result length: {len(json_str)} chars", flush=True)
    print(f"  result first 200: {repr(json_str[:200])}", flush=True)
else:
    print("  result: None", flush=True)

# Step 2: Try _find_json_objects
print("Step 2: _find_json_objects...", flush=True)
t3 = time.time()
candidates = _find_json_objects(raw)
t4 = time.time()
print(f"  _find_json_objects: {t4 - t3:.3f}s, found {len(candidates)} candidates", flush=True)
for i, c in enumerate(candidates):
    print(f"  candidate {i}: {len(c)} chars", flush=True)

# Step 3: Try parse_prediction
print("Step 3: parse_prediction...", flush=True)
t5 = time.time()
result = parse_prediction(raw, sample_id="test", run_id="test")
t6 = time.time()
print(f"  parse_prediction: {t6 - t5:.3f}s", flush=True)
print(f"  Result: {result}", flush=True)
print(f"Total: {time.time() - t0:.3f}s", flush=True)
