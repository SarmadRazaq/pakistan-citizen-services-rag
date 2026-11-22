# Seed source list: draft for approval

**Status:** draft, 22 Nov 2026. **Decisions D1–D5 taken by the student on 22 Nov 2026** (see the end). Nothing has been downloaded into the project yet. Each URL below was opened during a read-only check on 22 Nov 2026 unless it is marked *unverified*. Fee amounts are deliberately not recorded here; they will come only from the collected files.

**How to read this:**
- **Use** is one of:
  - `core`: part of the main corpus and the golden set;
  - `pair`: an older or newer version kept for superseded-vs-current tests;
  - `test`: OCR and other edge-case tests only, never part of the eval corpus;
  - `excluded (D1)`: left out by decision D1 (see the decisions at the end).
- **Collect** is one of:
  - `script`: `collect_corpus.py` can fetch it politely;
  - `manual`: the site blocks scripts, so you save the file yourself in a browser. The collector will **not** try to get around a block.

## 1. Identity documents (NADRA, federal)

NADRA's CDN returns 403 to every script, so collection is manual for all NADRA items. Its terms prohibit "unauthorised … copying" of its information system and give no reuse licence. See decision D2.

| ID | Use | Title | URL | Format | Lang | Date shown | Notes | Collect |
|---|---|---|---|---|---|---|---|---|
| ID-01 | core | Fee Structure (CNIC, Smart NIC, CRC, FRC, NICOP, …) | https://www.nadra.gov.pk/feeStructure | HTML (JS) | en | only the site footer "Last Updated 17-09-2026"; no notification no. | About 20 tables (normal/urgent/executive). Each table appears twice in the page source, so remove duplicates. | manual |
| ID-02 | core | CNIC: new / renew / update | https://www.nadra.gov.pk/identityDocument/cnic?tab=nic&action=new (renew: `&action=renew`) | HTML (JS) | en | cites NIC Rules 2002 r.12–13 | Reprint/lost tab has no URL of its own (click it) | manual |
| ID-03 | core | CRC / Juvenile card (B-form) | https://www.nadra.gov.pk/identityDocument/juvenile?action=new | HTML (JS) | en | NIC Rules 2002 r.18 | | manual |
| ID-04 | core | Family Registration Certificate | https://www.nadra.gov.pk/identityDocument/certificates/frc?tab=frc | HTML (JS) | en | NIC Rules 2002 r.19 | | manual |
| ID-05 | core | Registration Policy 6.0.2 (English) | https://www.nadra.gov.pk/assests/downloadsPdf/registration-policy-6-0-2-english.pdf | PDF, 44 pp | en | **"Date of issue: 18 Sep 2026"** in every page header (the Downloads listing says 04-06-2025; use the header date) | text layer confirmed (read 22 Nov). Core policy. **Chapter "Change / Correction in Identity Documents" is pp. 12–16** (under NIC Rules 2002, Rule 13(1)): name, parents' info, date of birth, address, marital status, religion, gender, adding relatives, other fields. Requirements are two-column tables (resident vs non-resident; parents alive vs not; up to 5 vs over 5 years), a key table-extraction test. (The site really spells the path "assests".) | manual |
| ID-06 | core | Registration Policy 6.0.2 (Urdu) | https://www.nadra.gov.pk/assests/downloadsPdf/registration-policy-6-0-2-urdu.pdf | PDF | **ur** | published 20-10-2025 | **English/Urdu pair with ID-05**; Urdu extraction quality not yet tested | manual |
| ID-07 | pair | Old fee-structure page (archive.org snapshot) | https://web.archive.org/web/20250130132051/https://www.nadra.gov.pk/fee-structure/ | HTML | en | snapshot 30 Jan 2025 | superseded version of ID-01; third-party archive host, labelled as such | script |

## 2. Passport (DGIP, federal)

DGIP pages are plain HTML, have no robots.txt (404), respond fast and work with scripts. The terms reserve all IP rights; there is no reuse licence.

| ID | Use | Title | URL | Format | Lang | Date shown | Notes | Collect |
|---|---|---|---|---|---|---|---|---|
| PP-01 | core | General requirements for passport (documents, renewal, lost, priority, e-passport fee table) | https://dgip.gov.pk/passport/ordinary-passport.php | HTML | en | last modified 23 Oct 2026 | HTML fee and processing-time tables. **Contradicts itself on processing days (5/21 vs 4/10 working days)**, a real contradiction test. | script |
| PP-02 | core | FAQ | https://dgip.gov.pk/faq/faq.php | HTML | en | 23 Oct 2026 | renewal, modification, online renewal | script |
| PP-03 | core | Passport Rules 2021 (SRO 245(I)/2022) | https://dgip.gov.pk/downloads/PassportRules-2021.pdf | PDF 26 pp | en | Gazette 14 Feb 2022 | text layer; repeals the 1974 Rules | script |
| PP-04 | pair | Passport Rules 1974 | https://dgip.gov.pk/downloads/Passport%20Rules%201974.pdf | PDF 11 pp | en | 1974 | superseded by PP-03 | script |
| PP-05 | test | SRO 1139(I)/2023 Passport Amendment Rules | https://dgip.gov.pk/downloads/SRO-NO-01-2023-Passport.pdf | PDF 10 pp | en | 2023 | **scanned (0 text chars)**, OCR test | script |
| PP-06 | core (D4) | Fee notices (images): "Enhancement of MRP Fee" (effective 07-03-2024); old MRP fee chart; fast-track fee letter No. 6/23/83-Policy (6 May 2024) | https://dgip.gov.pk/assets/img/Revised-PassportFee-2024-03.jpg · https://dgip.gov.pk/assets/img/Passport-Fee-Chart.png · https://dgip.gov.pk/assets/img/Revised-FastTrackFee-5-24.jpg | JPG/PNG | en | dated | The only *dated* passport fee notices. They are images not linked from any current page, ingested through the JPG/PNG OCR path (D4). The **old chart → 2024 notice** pair is a superseded-vs-current test. Fallback if the image path slips: drop this row. | script |

## 3. Social protection

BISP's disclaimer says it does not consent to its data being copied or "transformed into … pdf, doc, txt" without approaching its office. **Decision D1: BISP is excluded** until written permission arrives. The request is drafted in `corpus/BISP_permission_request.md`, for you to send. If permission comes, SP-01 to SP-03 move to `core`. Social protection is covered by SP-04 to SP-06. The federal health programme site blocks scripts.

| ID | Use | Title | URL | Format | Lang | Date shown | Notes | Collect |
|---|---|---|---|---|---|---|---|---|
| SP-01 | excluded (D1) | BISP Kafaalat programme page | https://www.bisp.gov.pk/Detail/YTgzNjkxM2YtN2ViMC00MjA5LWI0MDMtNzM4ZWJmMGVlNzc5 | HTML | en | Board decision 23 Sep 2021 | eligibility (PMT cut-off, relaxations) | script |
| SP-02 | excluded (D1) | Kafaalat registration process (Eng/Urdu) | https://www.bisp.gov.pk/SiteImage/Misc/files/KifalatEngUrdu_2022.pdf | PDF 2 pp | en + ur | Sep 2022 | **Urdu page extracts garbled**, so it needs OCR | script |
| SP-03 | excluded (D1) | NSER registration process (Eng/Urdu) | https://www.bisp.gov.pk/SiteImage/Misc/files/NSEREngUrdu_2022.pdf | PDF 5 pp | en + ur | Sep 2022 | bilingual; Urdu partly garbled | script |
| SP-04 | core | How to use the PM Health (Sehat) card | https://www.pmhealthprogram.gov.pk/steps-sehat-card | HTML | en | none | 6 steps; inpatient only | manual |
| SP-05 | core | PM Health FAQs | https://www.pmhealthprogram.gov.pk/faqs | HTML | en | none | eligibility (ICT/AJK/GB), coverage, complaints. **Expand every answer before saving.** | manual |
| SP-06 | core | KP Sehat Card Plus coverage | https://sehatcardplus.gov.pk/coverage | HTML | en | none | the only detailed covered/excluded-treatment list; **jurisdiction = KP** | manual |

**Excluded (personal data):**
- BISP Grievance Redress Report 2014: it has an annex of beneficiary names with their CNICs and phone numbers.
- BISP "HBL camps agents and user data" XLSX files and the BISP "Directory" PDF.

## 4. Individual tax (FBR, federal)

FBR has no robots.txt (404). Its [Terms of Use](https://fbr.gov.pk/Term-of-Use/544) let residents copy pages unmodified for free, non-commercial information sharing. This is the clearest reuse permission found.

| ID | Use | Title | URL | Format | Lang | Date shown | Notes | Collect |
|---|---|---|---|---|---|---|---|---|
| TX-01 | core | Income tax due dates | https://fbr.gov.pk/categ/income-tax-due-dates/51147/40846/81148 | HTML | en | no year | deadline table (person / due date) | script |
| TX-02 | core | Active Taxpayer List: what it is · inclusion · how to check status (3 pages) | https://fbr.gov.pk/categ/active-taxpayer-list-income-tax/51147/30859/71167 · …/71168 · …/71169 | HTML | en | inclusion page uses TY2021 examples (**stale**) | explainers only; **never the ATL itself** | script |
| TX-03 | core | File income tax return (IRIS steps) | https://fbr.gov.pk/categ/file-income-tax-return/51147/80860/71158 | HTML | en | none | procedure | script |
| TX-04 | core | Income Tax Ordinance 2001, amended up to 30-06-2026 | https://download1.fbr.gov.pk/Docs/2026724177725705IncomeTaxOrdinanace2001.pdf | PDF **839 pp** | en | 30.06.2026 | **long-document test**; schedules with rate tables | script |
| TX-05 | core | Withholding Tax Rate Card TY2027 | https://download1.fbr.gov.pk/Docs/202681113864992WithholdingTaxRatesCard2027.pdf | PDF 18 pp | en | Finance Act 2026 | **table test** (ATL vs non-ATL) | script |
| TX-06 | pair | Withholding Tax Rate Card TY2026 | http://download1.fbr.gov.pk/Docs/20258181281745641WHT-RateCard.pdf | PDF 14 pp | en | Finance Act 2025 | the "Late Filer" column was later removed; good version-diff test | script |
| TX-07 | core | Income Tax Rules 2002, **Urdu** | https://download1.fbr.gov.pk/Docs/2019101111056493IncomeTaxRules2002-Urdu-Kazmi.pdf | PDF 338 pp | **ur** | amended to 10.02.2017 | **long Urdu document**; Unicode text with some broken ligatures | script |
| TX-08 | pair | Brochure-006 "Filing of Income Tax Return" | https://download1.fbr.gov.pk/Docs/201092149546468brochue006GuidelinesoffilingIncomeTaxReturn.pdf | PDF 7 pp | en | July 2005 | clear "who must file" Q&A but **20 years old**; stale-information test (E45) | script |
| TX-09 | test | Taxpayers' Charter | https://download1.fbr.gov.pk/Docs/20218611868449Taxpayercharterupdated.pdf | PDF 6 pp | en | Aug 2021 | scanned with an OCR layer | script |

## 5. Rights and complaints (federal)

| ID | Use | Title | URL | Format | Lang | Date shown | Notes | Collect |
|---|---|---|---|---|---|---|---|---|
| RC-01 | core | Constitution of Pakistan (English) | https://pakistancode.gov.pk/pdffiles/administrator9d8e2ecc414c6d3371ac41114b61a2c4.pdf | PDF 176 pp | en | includes the 27th Amendment Act 2025 | text layer. Pakistan Code says its texts are "not authentic" and refers to the Gazette, so cite it as "Pakistan Code (MoLJ)". Mirror: Senate copy https://www.senate.gov.pk/uploads/documents/1764658026_642.pdf | script (Crawl-delay respected) |
| RC-02 | core | Wafaqi Mohtasib: FAQs about complaints | https://mohtasib.gov.pk/Detail/YmZhYzY4ODktNjNkMC00ZWNlLWI4YjAtYjViYzJmZjlkYjc2 | HTML | **en + ur** | none | **best single page**: 3-month limit, no fee, 60-day resolution, review, exclusions | script |
| RC-03 | core | Wafaqi Mohtasib: making a complaint | https://mohtasib.gov.pk/Detail/ODE2MWM4M2QtNDM5NS00OTc2LThlZmEtYzQ0NzViYWFmMTgz | HTML | en + ur | none | how to file | script |
| RC-04 | core | Wafaqi Mohtasib: who can lodge a complaint | https://mohtasib.gov.pk/Detail/NDE1YTBmODYtN2MzMC00ZGU0LTlkOWYtNzNmZTg4OThmYTFi | HTML | en + ur | none | jurisdiction | script |
| RC-05 | core | Investigation & Disposal of Complaints Regulations 2013 | https://mohtasib.gov.pk/SiteImage/Downloads/disposal_regulation_2013.pdf | PDF 68 pp | en | as on 14-10-2013 | procedural; includes blank forms | script |
| RC-06 | core | Complaint Form 'A' (English) | https://mohtasib.gov.pk/SiteImage/Downloads/form_a_english.pdf | PDF 1 p | en | 2016 | blank form | script |
| RC-07 | core | Pakistan Citizen Portal: complaints handling manual 3.0 | https://admin.pmdu.gov.pk/manuals/Pakistan-Citizen-Portal-Manual-3.0.pdf | PDF 34 pp | en | Jan 2020 | written for government offices; escalation rules | script |
| RC-08 | pair | Citizen Portal manual 2.0 | http://federalseed.gov.pk/SiteImage/Misc/files/Pakistan-Citizen-Portal-Manual-2_0.pdf | PDF 30 pp | en | Jul 2019 | superseded by RC-07 | script |
| RC-09 | core | Ministry of Human Rights: helpline 1099 | https://mohr.gov.pk/Detail/YWRiYjU0NzktMGE2Zi00NDYyLTljNzktMTA2N2M2MjBlZWZl | HTML | en | stats to May 2025 | free legal advice and referral; source for `referrals.yaml`. **Its operating hours conflict across the site.** | script |
| RC-10 | test | Complaint Form 'A' (Urdu) | https://mohtasib.gov.pk/SiteImage/Downloads/form_a_urdu.pdf | PDF 1 p | ur | 2016 | **image-only**, small Urdu OCR test | script |
| RC-11 | test | Constitution (Urdu, NA, as amended to 31 May 2018) | https://na.gov.pk/uploads/documents/1550129957_614.pdf | PDF 257 pp | ur | 2018 | **scanned with a garbage Latin OCR layer**: tests detecting "text that exists but is junk" (E46) | script |
| RC-12 | test | WMS Urdu brochure | https://mohtasib.gov.pk/SiteImage/Downloads/WMS-Urdu%20Broucher_updated_16_08_2024.pdf | PDF 10 pp | ur | 16-08-2024 | text converted to **vector outlines** (no text, no images), so pages must be rendered then OCR'd | script |

**Not used:**
- The **current Urdu Constitution** (Pakistan Code, 263 pp): it is image-only and 102 MB, which is above the 50 MB upload cap. Its amendment level couldn't be confirmed without OCR. It can serve as the "very large file → 413" demo file in `corpus/adversarial/`.

**Excluded (personal data):**
- WMS Annual Report 2025: it names complainants.
- The Compendium 2026: it names people and includes a phone number.
- The Citizen Portal statistics report: its officer-wise table names officials.

## 6. Provincial: Punjab and Sindh

| ID | Use | Title | URL | Format | Lang | Date shown | Notes | Collect |
|---|---|---|---|---|---|---|---|---|
| PB-01 | core | Punjab Consumer Protection Act 2005 (consolidated) | https://punjabcode.punjab.gov.pk/uploads/articles/the_punjab_consumer_protection_act_2005-pdf.pdf | PDF 16 pp | en | updated for Amendment Act LIV of 2025 | consumer court = District/Additional District Judge (2025); appeal → Lahore High Court | script |
| PB-02 | core | Punjab Consumer Protection Rules 2009 | https://punjabcode.punjab.gov.pk/uploads/articles/punjab-consumer-protection-rules-2009-pdf.pdf | PDF 12 pp | en | 22 May 2009 | complaints to the District Consumer Protection Council | script |
| PB-03 | core | How to file a consumer claim / complaint (English) | https://consumerservice.punjab.gov.pk/uploads/guidelines/filing_case_complaint_eng.pdf | PDF 2 pp | en | Sep 2022 | **predates the 2025 change**, so it is superseded guidance vs current law (E10) | script |
| PB-04 | core | Same guideline (Urdu) | https://consumerservice.punjab.gov.pk/uploads/guidelines/filing_case_complaint_urdu.pdf | PDF 2 pp | **ur** | Sep 2022 | **clean Urdu text layer**; English/Urdu pair with PB-03 | script |
| PB-05 | core | Provincial Motor Vehicles Ordinance 1965 (Punjab, consolidated) | https://punjabcode.punjab.gov.pk/uploads/articles/provincial-motor-vehicles-ordinance-1965-doc-doc-pdf.pdf | PDF 132 pp | en | amended to 2025/26 | licensing (Ch. II); juvenile motorcycle permit from 16 | script |
| PB-06 | core | DLIMS fee structure | https://dlims.punjab.gov.pk/fee_structure | HTML tables | en | **no date** | licence fees by category; regular licence for 1–5 years | script |
| PB-07 | core | DLIMS home: steps, FAQ, documents | https://dlims.punjab.gov.pk/ | HTML | en | none | services moved to the Dastak platform | script |
| PB-08 | core | City Traffic Police Lahore: licence issuance | https://ctplahore.gop.pk/license-issuance | HTML | en | none | **fee figures conflict with PB-06**: two official sources disagree (E10) | script |
| PB-09 | test | Legal-notice guideline (Urdu) | https://consumerservice.punjab.gov.pk/uploads/guidelines/legal_notice_urdu_guidelines.pdf | PDF 2 pp | ur | none | **scanned, with a garbage OCR layer**; the best scanned-Urdu OCR test | script |
| PB-10 | test | Punjab Consumer Protection Act (Urdu) | https://punjabcode.punjab.gov.pk/uploads/articles/UR_THE_PUNJAB_CONSUMER_PROTECTION_ACT_2005..pdf | PDF 16 pp | ur | 2018 (pre-2025) | has a text layer, but **extracted Urdu is scrambled**; also outdated | script |
| SD-01 | core | Sindh Consumer Protection Act 2014 + Rules 2017 (consolidated, Sindh Judicial Academy) | https://sja.gos.pk/assets/Updated_Laws/Sindh%20Consumer%20Protection%20Act,%202014.pdf | PDF 35 pp | en | amended to Act XXIX of 2023 | **best Sindh item**: consumer court = Judicial Magistrate; Rule 13 form of claim | script |
| SD-02 | pair | Sindh Consumer Protection (Amendment) Act (Sindh Act XXIX of 2023), text version | https://www.sindhlaws.gov.pk/Setup/Publications_SindhCode/PUB-NEW-25-000078.pdf | PDF 2 pp | en | Gazette 11 Aug 2023 | appeals → District & Sessions Court | script |
| SD-03 | pair | Sindh Consumer Protection Act (original, Sindh Code text) | https://www.sindhlaws.gov.pk/Setup/Publications_SindhCode/PUB-15-000096.pdf | PDF 23 pp | en | 2015 | the unamended version, superseded by SD-01 | script |
| SD-04 | core | Sindh driving licence fee structure | https://dls.sindhpolice.gov.pk/fees/ | HTML | en | **no date** | validity 3 or 5 years; the table has glitches ("Nill", empty cells) | script |
| SD-05 | test | Sindh Consumer Protection Act 2015 Gazette (scanned) | https://www.sindhlaws.gov.pk/GazetteDownload.aspx?PubID=PUB-15-000081 | PDF 23 pp | en | Gazette 18 Mar 2015 | poor scan with bleed-through; OCR stress test | script |

**Confirmed Punjab vs Sindh differences (for E31 and the golden set):**
- **Consumer court:** a District Judge in Punjab; a Judicial Magistrate in Sindh.
- **Appeal:** the Lahore High Court in Punjab; the District & Sessions Court in Sindh.
- **Administrative complaint route:** Punjab has the District Council plus an online portal; none was found for Sindh.
- **Licence validity:** 1–5 years in Punjab; 3 or 5 years in Sindh.
- **Licence categories:** the lists differ between the provinces.

**Weak or missing:**
- **Sindh licence rules:** the Sindh Code documents for them are broken ("PDF file not found").
- **Sindh licence procedure:** the DLS site lists no required documents.
- **Sindh complaint guidance:** no official complaint guide was found.
- **Licence fees in both provinces:** neither shows a date or notification number.

**Excluded (personal data):**
- Punjab consumer court cause list.
- Sindh High Court consumer judgments.
- Sindh Police absconder and watch lists.

## 7. What the check changed

1. **No on-topic DOCX or PPTX exists on any of these sites.** The only DOCX files found are off-topic (job proformas, an award letter, BISP project documents). The brief requires DOCX, so the plan is:
   - **Derived DOCX** from FBR pages (TX-01, TX-02). FBR's terms allow unmodified copies; the files are labelled `derived` in the manifest.
   - **A short helpdesk guidance note written by you** in DOCX, used as the demo *private upload*. It cites the official sources and fits the multi-tenancy story (E35).
   - **PPTX:** a few awareness slides you write yourself from cited sources, labelled `authored`. They are not presented as official material.
2. **No text-searchable official Urdu Constitution exists**, so it can't anchor cross-lingual questions. The English–Urdu question pairs will use **NADRA Registration Policy (ID-05/06)**, the **Mohtasib bilingual pages (RC-02–04)** and the **Punjab consumer guideline (PB-03/04)** instead.
3. **Urdu extraction often fails in ways that look like success.** It returns junk Latin characters, scrambled Urdu, or no text at all (vector outlines). The ingestion pipeline needs a **text-quality check**, not just "is there text": a script ratio, a dictionary hit rate, and OCR as the fallback. This becomes edge case E46.
4. **Tesseract is not installed on this machine.** OCR needs Tesseract with the `urd` and `eng` language data. It is a setup step for you.
5. **Real contradictions found:**
   - DGIP's own processing days (PP-01);
   - DLIMS vs City Traffic Police Lahore fees (PB-06/08);
   - the 2022 Punjab guideline vs the 2025 Act (PB-03 vs PB-01);
   - old vs new versions: Passport Rules 1974 → 2021, WHT card TY2026 → TY2027, Citizen Portal manual 2.0 → 3.0, NADRA fee page Jan 2025 → current.
6. **Coverage:** about 34 core and pair items, plus 8 test-only items.

## Decisions

| # | Decision | Status | What it means |
|---|---|---|---|
| D1 | **BISP excluded for now**; ask for permission in parallel | **decided 22 Nov** | SP-01 to SP-03 are excluded. `corpus/BISP_permission_request.md` is a draft email **for you to send** from your own address. If BISP grants permission in writing, the rows move to `core` and the reply is kept with the manifest. |
| D2 | **NADRA collected manually** by you; files used locally, never committed | **decided 22 Nov** | Follow `corpus/MANUAL_COLLECTION.md` for ID-01 to ID-06. The manifest records URL, retrieval date and SHA-256. |
| D3 | **PM Health and KP Sehat collected manually**, same as D2 | **decided 22 Nov** | SP-04 to SP-06, same checklist |
| D4 | **JPG/PNG ingestion added** (image → OCR), if time allows | **decided 22 Nov** | PP-06 moves to `core`. The blueprint adds the image extractor, magic-byte validation and an upload type. It is cut before anything mandatory if time runs short; PP-06 is then dropped. |
| D5 | Keep raw files out of git; commit only `manifest.csv`, the script and this list | **decided 22 Nov** | `corpus/raw/` and `corpus/adversarial/raw/` are in `.gitignore`; `make corpus` rebuilds them from the manifest, and SHA-256 detects changes. None of these sites grants a redistribution licence. |
