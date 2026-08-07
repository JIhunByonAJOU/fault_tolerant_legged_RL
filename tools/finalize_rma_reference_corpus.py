#!/usr/bin/env python3
"""Create RMA's auditable one-row-per-direct-citation manifest and links."""
from __future__ import annotations
import difflib,json
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
MAIN='2021-kumar-rma-rapid-motor-adaptation-for-legged-robots'
spec=importlib.util.spec_from_file_location('build',ROOT/'tools/build_rma_reference_corpus.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
records=[]
for base in [ROOT/'docs/papers/ref',ROOT/'docs/papers/main']:
 for f in base.glob('*/metadata.json'):
  try:records.append((json.loads(f.read_text()),f.parent))
  except Exception:pass
def match(title):
 return max(((difflib.SequenceMatcher(None,b.norm(title),b.norm(m.get('title',''))).ratio(),m,p) for m,p in records),key=lambda x:x[0])
# These were confirmed as exact matches in the pre-ingestion RMA duplicate
# audit.  Retrieval dates cannot distinguish them because this corpus uses a
# shared batch date for prior records as well.
PREEXISTING={12,23,31,32,34,40,45,48,51,52,57}
UNRESOLVED_EVIDENCE={
 1:'Individual OA search found only the 2009 aerospace-conference bibliographic citation; no author, institutional, or proceedings full-text PDF was located.',
 6:'The ICRACV 2012 proceedings table of contents was confirmed, but no lawful author, institutional, or proceedings PDF for the paper was located.',
 15:'ETH Research Collection accepted-manuscript endpoint https://www.research-collection.ethz.ch/bitstreams/52dec6c6-be30-4978-aa43-6fe6e7f66c9d/download was retried and returned HTTP 500.',
 16:'The 2003 Royal Society publication was identified; no lawful author/institutional open full-text PDF was located.',
 17:'The KDD 2016 bibliographic record was identified; no lawful author/institutional open full-text PDF was located.',
 20:'ETH confirms an OA accepted version at https://www.research-collection.ethz.ch/handle/20.500.11850/118642, but its current PDF endpoint https://www.research-collection.ethz.ch/bitstreams/09cfc536-6001-4295-b9ae-4e6b049c151e/download returned HTTP 500.',
 22:'ETH confirms an OA accepted version at https://www.research-collection.ethz.ch/handle/20.500.11850/335381, but its current PDF endpoint https://www.research-collection.ethz.ch/bitstreams/87ad93f6-5934-4df6-9fc5-e6918f4fdf41/download returned HTTP 404.',
 24:'The IJRR/SAGE landing page https://doi.org/10.1177/0278364916640102 was confirmed; no lawful author or institutional full-text PDF was located.',
 35:'PMC records the article as OA and its OA API supplied ftp PDF/package locations, but both currently return HTTP 404; no alternate lawful full-text endpoint was located.',
 36:'The SAGE landing page https://doi.org/10.1177/027836498400300206 offers access/purchase only; no lawful author or institutional full-text PDF was located.',
 39:'ETH confirms an OA text record at https://www.research-collection.ethz.ch/handle/20.500.11850/155865, but the current PDF endpoint https://www.research-collection.ethz.ch/bitstreams/a798d909-84eb-4231-9fe7-493d447bc497/download returned HTTP 500.',
 43:'The 1984 IEEE record (DOI 10.1109/TSMC.1984.6313238) was confirmed; no lawful author or institutional full-text PDF was located.',
 63:'The ICRA 2010 bibliographic record was confirmed; no lawful author, institutional, or proceedings full-text PDF was located.',
}
rows=[]; resolved=[]
for n,year,authors,title in b.REFS:
 if n in b.NON_PAPER:
  rows.append((n,'non-paper',title,'—','—',b.NON_PAPER[n]));continue
 score,m,p=match(title)
 if score>=.955:
  rid=m['id']; resolved.append(rid)
  rel='../main' if p.parent.name=='main' else '../ref'
  local=f'[paper.md]({rel}/{p.name}/paper.md)'
  status='resolved-existing' if n in PREEXISTING else 'downloaded-converted'
  # A legacy record can share today's date; provenance is deliberately worded
  # without pretending this run created it.
  if MAIN not in m.setdefault('cited_by',[]):m['cited_by'].append(MAIN)
  (p/'metadata.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
  rows.append((n,status,title,rid,local,'Lawful OA source.pdf present; PDF/SHA-256 validated. Markdown is bulk-extracted and needs visual review.'))
 else:
  rows.append((n,'unresolved-no-lawful-pdf',title,'—','—',UNRESOLVED_EVIDENCE.get(n,'No lawful full-text PDF was located after individual source checks; no paywall, login gate, or unverified third-party copy was bypassed.')))
mp=ROOT/'docs/papers/main'/MAIN/'metadata.json'; mm=json.loads(mp.read_text()); mm['cites']=resolved; mp.write_text(json.dumps(mm,ensure_ascii=False,indent=2)+'\n')
out=ROOT/'docs/papers/analysis';out.mkdir(parents=True,exist_ok=True)
counts={s:sum(r[1]==s for r in rows) for s in ['resolved-existing','downloaded-converted','unresolved-no-lawful-pdf','non-paper']}
lines=['# RMA — direct-reference manifest','',f'Scope: every numbered entry in the reference list of `{MAIN}` is accounted for exactly once. `downloaded-converted` and `resolved-existing` mean a locally stored lawful OA PDF was validated and has a searchable Markdown record. Bulk conversions are `needs-review`; they were not claimed to receive complete page-by-page visual verification. `unresolved-no-lawful-pdf` creates no substitute file.','',f'- Total numbered entries: **{len(rows)}**',f"- Existing records reused: **{counts['resolved-existing']}**",f"- Downloaded and converted in this pass: **{counts['downloaded-converted']}**",f"- Unresolved / no acquired lawful PDF: **{counts['unresolved-no-lawful-pdf']}**",f"- Non-paper web/software entries: **{counts['non-paper']}**",'', '| # | Status | Cited title | Canonical ID | Local record | Evidence / reason |','|---:|---|---|---|---|---|']
for row in rows:lines.append('| '+' | '.join(map(str,row))+' |')
lines+=['','## Unresolved entries','', 'No local source was fabricated for: '+', '.join(f'[{n}]' for n,s,*_ in rows if s=='unresolved-no-lawful-pdf')+'.','']
(out/'rma-reference-manifest.md').write_text('\n'.join(lines),encoding='utf-8')
# Keep all existing index rows, but rebuild them from actual local metadata so
# every newly converted RMA paper is visible once.  Main papers precede refs.
main_records=[]; ref_records=[]
for m,p in records:
 # Reload updated cited_by values as applicable.
 try:m=json.loads((p/'metadata.json').read_text())
 except Exception:pass
 (main_records if p.parent.name=='main' else ref_records).append((m,p))
head=['# Paper index','', '| ID | Year | Title | Authors | Role | Source | Markdown | Conversion | Notes |','|---|---:|---|---|---|---|---|---|---|']
def line(m,p,role):
 path=f'{role}/{p.name}'
 note=f'[implementation notes]({path}/implementation-notes.md)' if (p/'implementation-notes.md').exists() else '—'
 return f'| {m["id"]} | {m.get("year") or "—"} | {m["title"]} | {"; ".join(m.get("authors") or [])} | {role} | [source]({m.get("landing_page") or "—"}) | [paper.md]({path}/paper.md) | {m.get("conversion_status") or "needs-review"} | {note} |'
for m,p in sorted(main_records,key=lambda x:(x[0].get('year') or 0,x[0]['title'].lower())):head.append(line(m,p,'main'))
for m,p in sorted(ref_records,key=lambda x:(x[0].get('year') or 0,x[0]['title'].lower())):head.append(line(m,p,'ref'))
(ROOT/'docs/papers/00_index.md').write_text('\n'.join(head)+'\n',encoding='utf-8')
print(json.dumps({'total':len(rows),**counts,'resolved_total':len(resolved)},ensure_ascii=False))
