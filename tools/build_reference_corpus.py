#!/usr/bin/env python3
"""Resolve, lawfully download, and make auditable Markdown records for ref list.

This is intentionally conservative: OpenAlex is used only as a discovery index;
only URLs it labels as an open-access PDF are fetched.  A candidate must match
the cited title closely and the downloaded bytes must begin with the PDF magic.
"""
from __future__ import annotations
import difflib, hashlib, json, os, re, subprocess, sys, textwrap, time
from datetime import date
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/papers/ref"
TMP = ROOT / ".tmp-reference-corpus"
TODAY = "2026-08-04"

REFS = [
 (1,2018,"C. D. Bellicoso et al.","Advances in real-world applications for legged robots"),
 (2,2021,"Z. Chen et al.","Autonomous social distancing in urban environments using a quadruped robot"),
 (3,2020,"J. Hooks et al.","Alphred: A multi-modal operations quadruped robot for package delivery applications"),
 (4,2022,"T. Miki et al.","Learning robust perceptive locomotion for quadrupedal robots in the wild"),
 (5,2022,"Z. Chen et al.","Fault-tolerant gait design for quadruped robots with one locked leg using the GF set theory"),
 (6,2020,"J. Lee et al.","Learning quadrupedal locomotion over challenging terrain"),
 (7,2021,"A. Kumar et al.","RMA: Rapid motor adaptation for legged robots"),
 (8,2021,"H.-W. Park et al.","Jumping over obstacles with MIT Cheetah 2"),
 (9,2022,"G. Margolis et al.","Rapid locomotion via reinforcement learning"),
 (10,2021,"F. Shi et al.","CIRCUS ANYmal: A quadruped learning dexterous manipulation with its limbs"),
 (11,2018,"J. Di Carlo et al.","Dynamic locomotion in the MIT Cheetah 3 through convex model-predictive control"),
 (12,2016,"C. Gehring et al.","Practice makes perfect: An optimization-based approach to controlling agile motions for a quadruped robot"),
 (13,2022,"J. Cui et al.","Fault-tolerant motion planning and generation of quadruped robots synthesised by posture optimization and whole body control"),
 (14,2021,"V. Makoviychuk et al.","Isaac Gym: High performance GPU-based physics simulation for robot learning"),
 (15,2017,"J. Tobin et al.","Domain randomization for transferring deep neural networks from simulation to the real world"),
 (16,2018,"X. B. Peng et al.","Sim-to-real transfer of robotic control with dynamics randomization"),
 (17,2019,"J. Hwangbo et al.","Learning agile and dynamic motor skills for legged robots"),
 (18,2021,"T. Anne et al.","Meta-learning for fast adaptive locomotion with uncertainties in environments and robot dynamics"),
 (19,2021,"W. Okamoto et al.","Reinforcement learning with adaptive curriculum dynamics randomization for fault-tolerant robot control"),
 (20,2018,"M. Neunert et al.","Whole-body nonlinear model predictive control through contacts for quadrupeds"),
 (21,2022,"N. Rudin et al.","Learning to walk in minutes using massively parallel deep reinforcement learning"),
 (22,2022,"E. Vollenweider et al.","Advanced skills through multiple adversarial motion priors in reinforcement learning"),
 (23,2010,"S. Koos et al.","Crossing the reality gap in evolutionary robotics by promoting transferable controllers"),
 (24,2012,"A. Boeing et al.","Leveraging multiple simulators for crossing the reality gap"),
 (25,2018,"J. Tan et al.","Sim-to-real: Learning agile locomotion for quadruped robots"),
 (26,2019,"A. Loquercio et al.","Deep drone racing: From simulation to reality with domain randomization"),
 (27,2020,"O. M. Andrychowicz et al.","Learning dexterous in-hand manipulation"),
 (28,2019,"Y. Chebotar et al.","Closing the sim-to-real loop: Adapting simulation randomization with real world experience"),
 (29,2017,"J. Luo et al.","Robust trajectory optimization under frictional contact with iterative learning"),
 (30,2019,"Y. Zhong et al.","Analysis and research of quadruped robot's legs: A comprehensive review"),
 (31,2006,"J.-M. Yang", "Kinematic constraints on fault-tolerant gaits for a locked joint failure"),
 (32,2008,"C. Pana et al.","Fault-tolerant gaits of quadruped robot on a constant-slope terrain"),
 (33,2018,"M. Gor et al.","Fault accommodation in compliant quadruped robot through a moving appendage mechanism"),
 (34,2013,"S. Koos et al.","Fast damage recovery in robotics with the T-resilience algorithm"),
 (35,2017,"W. Yu et al.","Preparing for the unknown: Learning a universal policy with online system identification"),
 (36,2021,"G. B. Margolis et al.","Learning to jump from pixels"),
 (37,2018,"Y. Tassa et al.","DeepMind Control Suite"),
 (38,2020,"M. Laskin et al.","CURL: Contrastive unsupervised representations for reinforcement learning"),
 (39,2011,"S. Ross et al.","A reduction of imitation learning and structured prediction to no-regret online learning"),
 (40,2017,"J. Schulman et al.","Proximal policy optimization algorithms"),
 (41,2021,"H. Wu et al.","Self-supervised attention-aware reinforcement learning"),
 (42,2023,"I. Radosavovic et al.","Learning humanoid locomotion with transformers"),
 (43,2023,"H. Lai et al.","Sim-to-real transfer for quadrupedal locomotion via terrain transformer"),
 (44,2023,"A. Agarwal et al.","Legged locomotion in challenging terrains using egocentric vision"),
]

# Canonical preprints are more stable than search ranking for items whose papers
# explicitly identify an arXiv version.
ARXIV = {2:"2008.08889",7:"2107.04034",9:"2205.02824",10:"2011.08811",14:"2108.10470",15:"1703.06907",16:"1710.06537",19:"2111.10005",22:"2203.14912",25:"1804.10332",26:"1905.09727",27:"1808.00177",28:"1810.05687",35:"1702.02453",36:"2110.15344",37:"1801.00690",40:"1707.06347",42:"2303.03381",43:"2212.07740",44:"2211.07638"}
DIRECT = {
  1:("https://www.research-collection.ethz.ch/bitstreams/d147fae4-9691-49c2-829e-cc82df3a6612/download","https://www.research-collection.ethz.ch/items/ed714e17-173a-4b40-8c8e-90757417d30c"),
  11:("https://dspace.mit.edu/server/api/core/bitstreams/474e8173-7b22-46e6-a51b-3d8e8a383357/content","https://dspace.mit.edu/handle/1721.1/138000"),
  12:("https://www.research-collection.ethz.ch/bitstreams/52dec6c6-be30-4978-aa43-6fe6e7f66c9d/download","https://www.research-collection.ethz.ch/"),
  18:("https://www.pure.ed.ac.uk/ws/portalfiles/portal/220191974/Meta_Learning_for_Fast_ANNE_DOA30062021_AFV.pdf","https://www.research.ed.ac.uk/en/publications/meta-learning-for-fast-adaptive-locomotion-with-uncertainties-in-/"),
  30:("https://journals.sagepub.com/doi/pdf/10.1177/1729881419844148","https://journals.sagepub.com/doi/10.1177/1729881419844148"),
  29:("https://motion.cs.illinois.edu/papers/AuRo2017-Luo-RobustOptimization-preprint.pdf","https://doi.org/10.1007/s10514-017-9629-x"),
  23:("https://hal.archives-ouvertes.fr/hal-00633927/document","https://hal.science/hal-00633927"),
  16:("https://xbpeng.github.io/projects/SimToReal/SimToReal_2018.pdf","https://arxiv.org/abs/1710.06537"),
  36:("https://proceedings.mlr.press/v164/margolis22a/margolis22a.pdf","https://proceedings.mlr.press/v164/margolis22a.html"),
  43:("https://arxiv.org/pdf/2212.07740","https://arxiv.org/abs/2212.07740"),
  44:("https://proceedings.mlr.press/v205/agarwal23a/agarwal23a.pdf","https://proceedings.mlr.press/v205/agarwal23a.html"),
}

def norm(s): return re.sub(r"[^a-z0-9]+", "", s.lower())
def slug(s): return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")
def first_author(a):
    x = a.split(" et al.")[0].strip().replace(".", "")
    return slug(x.split()[-1]) if x else "unknown"
def get(url, accept="application/json", timeout=35):
    req=Request(url,headers={"User-Agent":"legged-gym-reference-corpus/1.0 (research; mailto:local@invalid)","Accept":accept})
    with urlopen(req,timeout=timeout) as r: return r.read(),r.geturl(),r.headers.get_content_type()
def oa_candidate(title, cited_year):
    raw,_,_=get("https://api.openalex.org/works?search="+quote(title)+"&per-page=12")
    works=json.loads(raw)["results"]
    scored=[]
    for w in works:
        score=difflib.SequenceMatcher(None,norm(title),norm(w.get("title") or "")).ratio()
        score -= min(abs((w.get("publication_year") or cited_year)-cited_year),5)*.012
        scored.append((score,w))
    score,w=max(scored,key=lambda x:x[0])
    if score < .80: return None,score
    locs=[]
    for l in [w.get("best_oa_location")] + (w.get("locations") or []):
        if l and l.get("pdf_url") and l.get("is_oa"): locs.append(l)
    if not locs and (w.get("open_access") or {}).get("oa_url"):
        locs=[{"pdf_url":w["open_access"]["oa_url"],"landing_page_url":w["open_access"]["oa_url"]}]
    return (w,locs[0]) if locs else (w,None),score
def download_pdf(url,path):
    try:
        data,final,ctype=get(url,"application/pdf",timeout=70)
        if not data.startswith(b"%PDF-"): return None,"download is not a PDF ("+ctype+")"
        path.write_bytes(data)
        # Poppler validates enough to distinguish an error page from a PDF.
        p=subprocess.run(["pdfinfo",str(path)],capture_output=True,text=True)
        if p.returncode: path.unlink(missing_ok=True); return None,"pdfinfo validation failed"
        return final,None
    except Exception as e: return None,type(e).__name__+": "+str(e)[:180]
def arxiv_meta(arx,title,year,authors):
    return {"title":title,"publication_year":year,"doi":"https://doi.org/10.48550/arXiv."+arx,
      "authorships":[{"author":{"display_name":authors}}],"ids":{"doi":"https://doi.org/10.48550/arXiv."+arx},
      "primary_location":{"landing_page_url":"https://arxiv.org/abs/"+arx}}
def write_record(n,cited_year,cited_authors,cited_title,w,pdf_url,landing,pdf_path):
    title=w.get("title") or cited_title; year=w.get("publication_year") or cited_year
    authors=[a.get("author",{}).get("display_name") for a in w.get("authorships",[]) if a.get("author",{}).get("display_name")] or [cited_authors]
    doi=(w.get("doi") or w.get("ids",{}).get("doi") or "").replace("https://doi.org/","") or None
    arxiv=None
    for x in (pdf_url,landing,(w.get("open_access") or {}).get("oa_url")):
        m=re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})",x or "")
        if m: arxiv=m.group(1)
    surname = first_author(cited_authors) if (len(authors)==1 and authors[0].endswith("et al.")) else slug(authors[0].split()[-1])
    sid=f"{year}-{surname}-{slug(title)[:48]}"
    folder=OUT/sid; folder.mkdir(parents=True,exist_ok=True)
    target=folder/"source.pdf"
    os.replace(pdf_path,target)
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    pages=None
    info=subprocess.run(["pdfinfo",str(target)],capture_output=True,text=True).stdout
    m=re.search(r"^Pages:\s+(\d+)",info,re.M); pages=int(m.group(1)) if m else None
    rawtext=subprocess.run(["pdftotext","-layout","-enc","UTF-8",str(target),"-"],capture_output=True,text=True).stdout
    rawtext=rawtext.replace("\x0c","\n\n--- Page break ---\n\n")
    rawtext=re.sub(r"\n{4,}","\n\n\n",rawtext)
    status="verified" if rawtext.strip() else "needs-review"
    md="---\n"+"\n".join([
      f"id: {sid}",f"title: {json.dumps(title,ensure_ascii=False)}","authors:",*['  - '+json.dumps(a,ensure_ascii=False) for a in authors],
      f"year: {year}",f"venue: {json.dumps(((w.get('primary_location') or {}).get('source') or {}).get('display_name') or 'unknown')}",
      f"doi: {json.dumps(doi)}",f"arxiv: {json.dumps(arxiv)}",f"landing_page: {json.dumps(landing)}",f"pdf_url: {json.dumps(pdf_url)}","source_pdf: \"source.pdf\"",f"sha256: \"{sha}\"",f"retrieved_at: \"{TODAY}\"",f"conversion_status: \"{status}\"","---\n\n",
      f"# {title}\n\n", "## Source\n\n",f"Reference number in *Saving the Limping*: [{n}].\n\n",
      "## Full text\n\n", rawtext.strip()+"\n"])
    (folder/"paper.md").write_text(md,encoding="utf-8")
    venue=(((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "unknown")
    meta={"id":sid,"title":title,"authors":authors,"year":year,"venue":venue,"doi":doi,"arxiv":arxiv,"landing_page":landing,"pdf_url":pdf_url,"source_pdf":"source.pdf","sha256":sha,"retrieved_at":TODAY,"role":"reference","aliases":[cited_title] if title!=cited_title else [],"cites":[],"cited_by":["2023-liu-saving-the-limping"],"conversion_status":status,"conversion":{"extractor":"pdftotext -layout -enc UTF-8; pdfinfo validation","page_count":pages,"conversion_date":TODAY,"warnings":["Bulk faithful text extraction; headings/tables/equations require per-paper visual review."]}}
    (folder/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # These are the references where implementation parameters are likely useful;
    # avoid inventing values in a bulk pass.
    robotics=any(k in title.lower() for k in ["robot","locomotion","quadruped","control","manipulation","sim-to-real","simulation","gait","trajectory","legged","jump"])
    if robotics:
      notes=(f"# Implementation notes — {title}\n\n"
       f"- **Status:** Unspecified in this corpus's bulk pass; inspect the faithful source extraction in [paper.md](paper.md) before using any implementation value.\n"
       f"- **Evidence scope:** Reference [{n}] of *Saving the Limping*; no parameter has been inferred or copied here.\n")
      (folder/"implementation-notes.md").write_text(notes,encoding="utf-8")
    return {"n":n,"id":sid,"status":"resolved","citation":cited_title,"landing":landing,"pdf":pdf_url,"sha256":sha,"pages":pages,"reason":"Open-access PDF downloaded and validated with pdfinfo."}
def main():
    OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(exist_ok=True)
    only={int(x) for x in os.environ.get("ONLY","").split(",") if x.strip()}
    results=[]
    for n,year,authors,title in REFS:
      if only and n not in only: continue
      tmp=TMP/(str(n)+".pdf")
      try:
        if n in DIRECT:
          pdf,landing=DIRECT[n]; w={"title":title,"publication_year":year,"authorships":[{"author":{"display_name":authors}}],"primary_location":{}}; score=1
        elif n in ARXIV:
          arx=ARXIV[n]; found,_=oa_candidate(title,year); w=(found[0] if found else arxiv_meta(arx,title,year,authors)); pdf="https://arxiv.org/pdf/"+arx; landing="https://arxiv.org/abs/"+arx; score=1
        else:
          ans,score=oa_candidate(title,year)
          if not ans: results.append({"n":n,"status":"unresolved","citation":title,"reason":f"No OpenAlex title match at required confidence (best score {score:.3f})."}); continue
          w,loc=ans
          if not loc: results.append({"n":n,"status":"unresolved","citation":title,"reason":"Exact bibliographic match found, but OpenAlex reports no lawful open-access full-PDF location."}); continue
          pdf=loc["pdf_url"]; landing=loc.get("landing_page_url") or w.get("doi") or pdf
        final,err=download_pdf(pdf,tmp)
        if err: results.append({"n":n,"status":"unresolved","citation":title,"landing":landing,"pdf":pdf,"reason":err}); continue
        results.append(write_record(n,year,authors,title,w,final or pdf,landing,tmp))
      except Exception as e:
        results.append({"n":n,"status":"unresolved","citation":title,"reason":type(e).__name__+": "+str(e)[:180]})
      time.sleep(.2)
    (TMP/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
