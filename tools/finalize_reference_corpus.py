#!/usr/bin/env python3
"""Build the one-row-per-citation manifest and repository cross-links."""
import difflib, importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("builder",ROOT/"tools/build_reference_corpus.py")
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
metas=[]
for f in (ROOT/"docs/papers/ref").glob("*/metadata.json"):
    try:
        m=json.loads(f.read_text()); metas.append((m,f.parent))
    except Exception: pass
unresolved={
1:"Author PDF URL and ETH OA landing were checked; supplied author URL returned 404. No replacement downloadable OA file found.",
3:"Publisher record exists; a ResearchGate record was found but no author/institutional full-text PDF accessible without relying on that service.",
5:"DOI 10.1016/j.mechmachtheory.2022.105069 resolves to a paywalled publisher article; no lawful OA full text located.",
8:"MIT Biomimetic Robotics metadata page confirms the citation, but exposes no downloadable OA PDF; candidate sources returned access denial.",
12:"ETH's accepted-manuscript OA download endpoint was identified, but returned HTTP 500 during repeated direct retrieval; no other downloadable OA copy was found.",
24:"Exact UWA/IEEE metadata was found, but no lawful OA full-PDF location surfaced.",
30:"SAGE marks the article CC BY and its OA PDF endpoint was checked, but the endpoint returned HTTP 403 to non-interactive download; no alternative repository PDF was found.",
31:"DOI 10.1007/s10846-006-9054-4 metadata confirmed; publisher/repository search found no lawful OA PDF.",
32:"DOI 10.1109/AQTR.2008.4588739 confirmed. ResearchGate advertises author uploads, but no direct public PDF could be retrieved without login.",
33:"DOI 10.1016/j.mechmachtheory.2017.10.011 confirmed. ResearchGate advertises author upload, but no direct public PDF could be retrieved without login.",
}
rows=[]; resolved=[]
for n,year,authors,title in b.REFS:
    score,m,path=max(((difflib.SequenceMatcher(None,b.norm(title),b.norm(m["title"])).ratio(),m,p) for m,p in metas), key=lambda x:x[0])
    if score >= .95:
        rid=m["id"]; resolved.append(rid)
        rows.append((n,"resolved",title,rid,f"[paper.md](../ref/{path.name}/paper.md)","Open-access PDF downloaded and PDF/SHA-256 validated; Markdown is bulk-extracted and needs visual review."))
    else:
        rows.append((n,"unresolved",title,"—","—",unresolved[n]))
out=ROOT/"docs/papers/analysis"; out.mkdir(parents=True,exist_ok=True)
lines=["# Saving the Limping — direct-reference manifest","", "Scope: every numbered entry in the reference list of `2023-liu-saving-the-limping`, accounted for exactly once. `resolved` means a lawful OA full PDF was downloaded, opened with `pdfinfo`, SHA-256 validated, and bulk-extracted to Markdown. Every bulk Markdown record is `needs-review`: its equations, tables, and section ordering were not each visually verified. `unresolved` means no file was fabricated or paywall/access control bypassed.","",f"- Total: **{len(rows)}**",f"- Resolved PDF/downloaded/bulk-extracted: **{len(resolved)}**",f"- Unresolved after lawful OA search: **{len(rows)-len(resolved)}**","", "| # | Status | Cited title | Resolved ref ID | Local record | Evidence / reason |","|---:|---|---|---|---|---|"]
for n,status,title,rid,local,evidence in rows:
    lines.append(f"| {n} | {status} | {title} | {rid} | {local} | {evidence} |")
lines.extend(["","## Unresolved entries","", "The following were intentionally not downloaded from paywalled, login-gated, or technically inaccessible endpoints: "+", ".join(f"[{n}]" for n,s,*_ in rows if s=="unresolved")+".",""])
(out/"saving-the-limping-reference-manifest.md").write_text("\n".join(lines),encoding="utf-8")
# Main-paper cross-links.
mp=ROOT/"docs/papers/main/2023-liu-saving-the-limping/metadata.json"; mm=json.loads(mp.read_text()); mm["cites"]=resolved; mp.write_text(json.dumps(mm,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
# Rebuild index from the authoritative resolved set, preserving the main row.
index=["# Paper index","","| ID | Year | Title | Authors | Role | Source | Markdown | Conversion | Notes |","|---|---:|---|---|---|---|---|---|---|"]
main=mm; index.append(f'| {main["id"]} | {main["year"]} | {main["title"]} | {"; ".join(main["authors"])} | main | [source]({main["landing_page"]}) | [paper.md](main/2023-liu-saving-the-limping/paper.md) | {main["conversion_status"]} | [implementation notes](main/2023-liu-saving-the-limping/implementation-notes.md) |')
rs=[]
for m,p in metas:
  if m["id"] in resolved: rs.append((m,p))
for m,p in sorted(rs,key=lambda x:(x[0].get("year") or 0,x[0]["title"].lower())):
  note="[implementation notes](ref/%s/implementation-notes.md)"%p.name if (p/"implementation-notes.md").exists() else "—"
  index.append(f'| {m["id"]} | {m.get("year") or "—"} | {m["title"]} | {"; ".join(m["authors"])} | reference | [source]({m["landing_page"]}) | [paper.md](ref/{p.name}/paper.md) | {m["conversion_status"]} | {note} |')
(ROOT/"docs/papers/00_index.md").write_text("\n".join(index)+"\n",encoding="utf-8")
print(json.dumps({"resolved":len(resolved),"unresolved":len(rows)-len(resolved),"ids":resolved},ensure_ascii=False))
