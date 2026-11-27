# Manual collection checklist (decisions D2 and D3)

These sites refuse automated access: NADRA, PM Health and KP Sehat. The collection script will skip them, and so will I. Driving a browser with automation to save them would be exactly the automated access they block. You collect them by hand in your **own normal browser**, which takes about 15 minutes.

**Where files go:** `corpus/raw/manual/`. This folder stays out of git (decision D5, recommended). Use the file names given below. `collect_corpus.py --register-manual` will then:
- compute each file's SHA-256;
- fill in the manifest (URL, retrieval date, agency, jurisdiction);
- mark the rows `collected_by=manual`.

**Before you start:**
- If a cookie or consent banner appears, choose the most restrictive option (Reject or Necessary only).
- Do not log in, and do not type a CNIC or phone number anywhere.
- If a page shows anyone's personal details, skip it and note that in the log below.

**How to save:**
- **HTML pages:** in Chrome or Edge, press **Ctrl+S**, then choose **"Webpage, Complete"**. This keeps the page as you see it, including content built by JavaScript. Delete the `_files` folder afterwards; only the `.html` is needed.
- **Collapsed FAQ answers or tabs:** **expand every answer, or click the tab**, before saving. Only what is visible is saved.
- **PDFs:** open the link and use the browser's download or save button. Keep the original file and don't print it to PDF.

## NADRA (D2)

| ✓ | ID | Open | Before saving | Save as |
|---|---|---|---|---|
| ☐ | ID-01 | https://www.nadra.gov.pk/feeStructure | scroll to the bottom so every table loads | `ID-01__nadra-fee-structure.html` |
| ☐ | ID-02a | https://www.nadra.gov.pk/identityDocument/cnic?tab=nic&action=new | expand the FAQ answers | `ID-02a__nadra-cnic-new.html` |
| ☐ | ID-02b | https://www.nadra.gov.pk/identityDocument/cnic?tab=nic&action=renew | expand the FAQ answers | `ID-02b__nadra-cnic-renew.html` |
| ☐ | ID-02c | same CNIC page, **click the Reprint / Lost tab** | expand the FAQ answers | `ID-02c__nadra-cnic-reprint-lost.html` |
| ☐ | ID-03 | https://www.nadra.gov.pk/identityDocument/juvenile?action=new | expand the FAQ answers | `ID-03__nadra-crc-juvenile.html` |
| ☐ | ID-04 | https://www.nadra.gov.pk/identityDocument/certificates/frc?tab=frc | expand the FAQ answers | `ID-04__nadra-frc.html` |
| ☐ | ID-05 | https://www.nadra.gov.pk/assests/downloadsPdf/registration-policy-6-0-2-english.pdf | — | `ID-05__nadra-registration-policy-6-0-2-en.pdf` |
| ☐ | ID-06 | https://www.nadra.gov.pk/assests/downloadsPdf/registration-policy-6-0-2-urdu.pdf | (about 7 MB) | `ID-06__nadra-registration-policy-6-0-2-ur.pdf` |

ID-07, the archive.org copy of the old fee page, is not on NADRA's servers. The script collects it.

## PM Health / Sehat card (D3)

| ✓ | ID | Open | Before saving | Save as |
|---|---|---|---|---|
| ☐ | SP-04 | https://www.pmhealthprogram.gov.pk/steps-sehat-card | — | `SP-04__pmhealth-steps.html` |
| ☐ | SP-05 | https://www.pmhealthprogram.gov.pk/faqs | **expand every question**, since the answers are hidden until clicked | `SP-05__pmhealth-faqs.html` |
| ☐ | SP-06 | https://sehatcardplus.gov.pk/coverage | scroll to the bottom (exclusion list) | `SP-06__kp-sehat-coverage.html` |

Use `www.pmhealthprogram.gov.pk`, not the bare domain: the bare domain failed its security (TLS) check.

## Log

Fill in one line per file. The register step reads this.

| ID | Saved on (date) | Notes (e.g. "FAQ had 12 answers", "skipped: showed personal data") |
|---|---|---|
| | | |
