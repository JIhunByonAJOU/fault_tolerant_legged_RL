#!/usr/bin/env python3
"""Lawfully acquire and bulk-convert direct references of RMA.

Only OpenAlex locations explicitly flagged open-access with a direct PDF URL
are used.  Existing records are matched first by normalized title (and are
never overwritten).  The output deliberately remains ``needs-review``:
layout text is faithful/searchable but has not received page-by-page visual
verification.
"""
from __future__ import annotations
import difflib, hashlib, json, os, re, shutil, subprocess, time
from datetime import date
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'docs/papers/ref'
MAIN_ID='2021-kumar-rma-rapid-motor-adaptation-for-legged-robots'
TODAY='2026-08-04'
# (number, year, authors-as-cited, title).  Transcribed from RMA pp. 8--10.
REFS=[
(1,2009,'MYM Ahmed; N Qin','Surrogate-based aerodynamic design optimization: Use of surrogates in aerodynamic design optimization'),
(2,2014,'Aaron D Ames; Kevin Galloway; Koushil Sreenath; Jessy W Grizzle','Rapidly exponentially stabilizing control lyapunov functions and hybrid zero dynamics'),
(3,2018,'Taylor Apgar; Patrick Clary; Kevin Green; Alan Fern; Jonathan W Hurst','Fast online trajectory optimization for the bipedal robot Cassie'),
(4,2018,'Monica Barragan; Nikolai Flowers; Aaron M Johnson','MiniRHex: A small, open-source, fully programmable walking hexapod'),
(5,2018,'Gerardo Bledt; Matthew J Powell; Benjamin Katz; Jared Di Carlo; Patrick M Wensing; Sangbae Kim','MIT Cheetah 3: Design and control of a robust, dynamic quadruped robot'),
(6,2012,'Adrian Boeing; Thomas Braeunl','Leveraging multiple simulators for crossing the reality gap'),
(7,2005,'Josh C Bongard; Hod Lipson','Nonlinear system identification using coevolution of models and tests'),
(8,2016,'Roberto Calandra; Andre Seyfarth; Jan Peters; Marc Peter Deisenroth','Bayesian optimization for learning gaits under uncertainty'),
(9,2018,'Krzysztof Choromanski; Atil Iscen; Vikas Sindhwani; Jie Tan; Erwin Coumans','Optimizing simulations with noise-tolerant structured exploration'),
(10,2019,'Ignasi Clavera; Anusha Nagabandi; Simin Liu; Ronald S Fearing; Pieter Abbeel; Sergey Levine; Chelsea Finn','Learning to adapt in dynamic, real-world environments through meta-reinforcement learning'),
(11,2010,'Martin De Lasa; Igor Mordatch; Aaron Hertzmann','Feature-based locomotion controllers'),
(12,2018,'Jared Di Carlo; Patrick M Wensing; Benjamin Katz; Gerardo Bledt; Sangbae Kim','Dynamic locomotion in the MIT Cheetah 3 through convex model-predictive control'),
(13,2017,'Chelsea Finn; Pieter Abbeel; Sergey Levine','Model-agnostic meta-learning for fast adaptation of deep networks'),
(14,2018,'Scott Fujimoto; Herke Hoof; David Meger','Addressing function approximation error in actor-critic methods'),
(15,2016,'Christian Gehring; Stelian Coros; Marco Hutter; C Dario Bellicoso; Huub Heijnen; Remo Diethelm; Michael Bloesch; Peter Fankhauser; Jemin Hwangbo; Mark Hoepflinger','Practice makes perfect: An optimization-based approach to controlling agile motions for a quadruped robot'),
(16,2003,'Hartmut Geyer; Andre Seyfarth; Reinhard Blickhan','Positive force feedback in bouncing gaits'),
(17,2016,'Xiaoxiao Guo; Wei Li; Francesco Iorio','Convolutional neural networks for steady flow approximation'),
(18,2019,'Tuomas Haarnoja; Sehoon Ha; Aurick Zhou; Jie Tan; George Tucker; Sergey Levine','Learning to walk via deep reinforcement learning'),
(19,2017,'Josiah Hanna; Peter Stone','Grounded action transformation for robot learning in simulation'),
(20,2016,'Marco Hutter; Christian Gehring; Dominic Jud; Andreas Lauber; C Dario Bellicoso; Vassilios Tsounis; Jemin Hwangbo; Karen Bodie; Peter Fankhauser; Michael Bloesch','ANYmal - a highly mobile and dynamic quadrupedal robot'),
(21,2021,'Jemin Hwangbo','RaisimGymTorch'),
(22,2018,'Jemin Hwangbo; Joonho Lee; Marco Hutter','Per-contact iteration method for solving contact dynamics'),
(23,2019,'Jemin Hwangbo; Joonho Lee; Alexey Dosovitskiy; Dario Bellicoso; Vassilios Tsounis; Vladlen Koltun; Marco Hutter','Learning agile and dynamic motor skills for legged robots'),
(24,2016,'Dong Jin Hyun; Jongwoo Lee; SangIn Park; Sangbae Kim','Implementation of trot-to-gallop transition and subsequent gallop on the MIT Cheetah I'),
(25,2018,'Atil Iscen; Ken Caluwaerts; Jie Tan; Tingnan Zhang; Erwin Coumans; Vikas Sindhwani; Vincent Vanhoucke','Policies modulating trajectory generators'),
(26,2012,'Aaron M Johnson; Thomas Libby; Evan Chang-Siu; Masayoshi Tomizuka; Robert J Full; Daniel E Koditschek','Tail assisted dynamic self righting'),
(27,2010,'Mrinal Kalakrishnan; Jonas Buchli; Peter Pastor; Michael Mistry; Stefan Schaal','Fast, robust quadruped locomotion over challenging terrain'),
(28,2017,'Mahdi Khoramshahi; Hamed Jalaly Bidgoly; Soroosh Shafiee; Ali Asaei; Auke Jan Ijspeert; Majid Nili Ahmadabadi','Piecewise linear spine for speed-energy efficiency trade-off in quadruped robots'),
(29,2015,'Diederik P Kingma; Jimmy Ba','Adam: A method for stochastic optimization'),
(30,2013,'Jens Kober; J Andrew Bagnell; Jan Peters','Reinforcement learning in robotics: A survey'),
(31,2010,'Sylvain Koos; Jean-Baptiste Mouret; Stephane Doncieux','Crossing the reality gap in evolutionary robotics by promoting transferable controllers'),
(32,2020,'Joonho Lee; Jemin Hwangbo; Lorenz Wellhausen; Vladlen Koltun; Marco Hutter','Learning quadrupedal locomotion over challenging terrain'),
(33,2016,'Timothy P Lillicrap; Jonathan J Hunt; Alexander Pritzel; Nicolas Heess; Tom Erez; Yuval Tassa; David Silver; Daan Wierstra','Continuous control with deep reinforcement learning'),
(34,2017,'Jingru Luo; Kris Hauser','Robust trajectory optimization under frictional contact with iterative learning'),
(35,2018,'Jonathan Samir Matthis; Jacob L Yates; Mary M Hayhoe','Gaze and the control of foot placement when walking in natural terrain'),
(36,1984,'Hirofumi Miura; Isao Shimoyama','Dynamic walk of a biped'),
(37,2016,'Volodymyr Mnih; Adria Puigdomenech Badia; Mehdi Mirza; Alex Graves; Timothy Lillicrap; Tim Harley; David Silver; Koray Kavukcuoglu','Asynchronous methods for deep reinforcement learning'),
(38,2020,'Ofir Nachum; Michael Ahn; Hugo Ponte; Shixiang Shane Gu; Vikash Kumar','Multi-agent manipulation via locomotion using hierarchical sim2real'),
(39,2017,'Michael Neunert; Thiago Boaventura; Jonas Buchli','Why off-the-shelf physics simulators fail in evaluating feedback controller performance: A case study for quadrupedal robots'),
(40,2018,'Xue Bin Peng; Marcin Andrychowicz; Wojciech Zaremba; Pieter Abbeel','Sim-to-real transfer of robotic control with dynamics randomization'),
(41,2020,'Xue Bin Peng; Erwin Coumans; Tingnan Zhang; Tsang-Wei Edward Lee; Jie Tan; Sergey Levine','Learning agile robotic locomotion skills by imitating animals'),
(42,2019,'Delyle T Polet; John EA Bertram','An inelastic quadrupedal model discovers four-beat walking, two-beat running, and pseudo-elastic actuation as energetically optimal'),
(43,1984,'Marc H Raibert','Hopping in legged systems—modeling and simulation for the two-dimensional one-legged case'),
(44,2009,'Nathan Ratliff; Matt Zucker; J Andrew Bagnell; Siddhartha Srinivasa','CHOMP: Gradient optimization techniques for efficient motion planning'),
(45,2011,'Stephane Ross; Geoffrey Gordon; Drew Bagnell','A reduction of imitation learning and structured prediction to no-regret online learning'),
(46,2001,'Uluc Saranli; Martin Buehler; Daniel E Koditschek','RHex: A simple and highly mobile hexapod robot'),
(47,2016,'John Schulman; Philipp Moritz; Sergey Levine; Michael I Jordan; Pieter Abbeel','High-dimensional continuous control using generalized advantage estimation'),
(48,2017,'John Schulman; Filip Wolski; Prafulla Dhariwal; Alec Radford; Oleg Klimov','Proximal policy optimization algorithms'),
(49,2020,'Xingyou Song; Yuxiang Yang; Krzysztof Choromanski; Ken Caluwaerts; Wenbo Gao; Chelsea Finn; Jie Tan','Rapidly adaptable legged robots via evolutionary meta-learning'),
(50,2011,'Koushil Sreenath; Hae-Won Park; Ioannis Poulakakis; Jessy W Grizzle','A compliant hybrid zero dynamics controller for stable, efficient and fast bipedal walking on MABEL'),
(51,2018,'Jie Tan; Tingnan Zhang; Erwin Coumans; Atil Iscen; Yunfei Bai; Danijar Hafner; Steven Bohez; Vincent Vanhoucke','Sim-to-real: Learning agile locomotion for quadruped robots'),
(52,2017,'Josh Tobin; Rachel Fong; Alex Ray; Jonas Schneider; Wojciech Zaremba; Pieter Abbeel','Domain randomization for transferring deep neural networks from simulation to the real world'),
(53,2021,'Xingxing Wang','Unitree Robotics'),
(54,2020,'Zhaoming Xie; Xingye Da; Michiel van de Panne; Buck Babich; Animesh Garg','Dynamics randomization revisited: A case study for quadrupedal locomotion'),
(55,2020,'Yuxiang Yang; Ken Caluwaerts; Atil Iscen; Tingnan Zhang; Jie Tan; Vikas Sindhwani','Data efficient reinforcement learning for legged robots'),
(56,2007,'KangKang Yin; Kevin Loken; Michiel Van de Panne','Simbicon: Simple biped locomotion control'),
(57,2017,'Wenhao Yu; Jie Tan; C Karen Liu; Greg Turk','Preparing for the unknown: Learning a universal policy with online system identification'),
(58,2018,'Wenhao Yu; C Karen Liu; Greg Turk','Policy transfer with strategy optimization'),
(59,2019,'Wenhao Yu; Visak C V Kumar; Greg Turk; C Karen Liu','Sim-to-real transfer for biped locomotion'),
(60,2020,'Wenhao Yu; Jie Tan; Yunfei Bai; Erwin Coumans; Sehoon Ha','Learning fast adaptation with meta strategy optimization'),
(61,2019,'Wenxuan Zhou; Lerrel Pinto; Abhinav Gupta','Environment probing interaction policies'),
(62,2011,'J Zico Kolter; Andrew Y Ng','The Stanford LittleDog: A learning and rapid replanning approach to quadruped locomotion'),
(63,2010,'Matt Zucker; J Andrew Bagnell; Christopher G Atkeson; James Kuffner','An optimization approach to rough terrain locomotion'),
(64,2011,'Matt Zucker; Nathan Ratliff; Martin Stolle; Joel Chestnutt; J Andrew Bagnell; Christopher G Atkeson; James Kuffner','Optimization and learning for rough terrain legged locomotion'),]
NON_PAPER={21:'Software library / project webpage, not a paper.',53:'Manufacturer website, not a paper.'}
ARXIV={13:'1703.03400',14:'1802.09477',18:'1812.11103',19:'1703.07373',23:'1901.08652',29:'1412.6980',32:'2010.11251',33:'1509.02971',37:'1602.01783',38:'1910.11779',40:'1710.06537',41:'1901.02690',47:'1506.02438',48:'1707.06347',49:'2008.12687',51:'1804.10332',52:'1703.06907',54:'2011.02404',55:'2009.14034',57:'1702.02453',58:'1809.01193',59:'1909.01555',60:'1911.02523',61:'1810.04723'}
# Direct, lawful author/institution/proceedings PDFs found during the second
# pass.  Keep landing and PDF URLs separate for auditability.
DIRECT={
 2:('https://web.eecs.umich.edu/~grizzle/papers/CFL_Robotics_2012TAC.pdf','https://web.eecs.umich.edu/~grizzle/papers/CFL_Robotics_2012TAC.pdf'),
 3:('https://www.roboticsproceedings.org/rss14/p54.pdf','https://www.roboticsproceedings.org/rss14/p54.html'),
 5:('https://dspace.mit.edu/server/api/core/bitstreams/b93369ab-d87a-4fcf-a1e0-3bd34be52761/content','https://hdl.handle.net/1721.1/126619'),
 8:('https://www.robertocalandra.com/papers/2016_calandra_amai.pdf','https://www.robertocalandra.com/papers/2016_calandra_amai.pdf'),
 9:('https://arxiv.org/pdf/1805.07831','https://arxiv.org/abs/1805.07831'),
 15:('https://www.research-collection.ethz.ch/bitstreams/52dec6c6-be30-4978-aa43-6fe6e7f66c9d/download','https://www.research-collection.ethz.ch/'),
 20:('https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/118642/1/eth-49454-01.pdf','https://www.research-collection.ethz.ch/handle/20.500.11850/118642'),
 22:('https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/335381/contact-iteration-method%20%284%29.pdf?sequence=1','https://www.research-collection.ethz.ch/handle/20.500.11850/335381'),
 30:('https://www.ri.cmu.edu/pub_files/2013/7/Kober_IJRR_2013.pdf','https://www.ri.cmu.edu/publications/reinforcement-learning-in-robotics-a-survey/'),
 44:('https://www.cs.cmu.edu/~mzucker/icra09-chomp.pdf','https://publications.ri.cmu.edu/chomp-gradient-optimization-techniques-for-efficient-motion-planning'),
 46:('https://www.rhex.web.tr/saranli_buehler_koditschek.ijrr2001.pdf','https://www.rhex.web.tr/'),
 56:('https://www.cs.sfu.ca/~kkyin/papers/Yin_SIG07.pdf','https://www.cs.ubc.ca/~van/papers/simbicon.htm'),
 26:('https://kodlab.seas.upenn.edu/uploads/Aaron/tails_final.pdf','https://kodlab.seas.upenn.edu/publications/'),
 27:('https://fileadmin.cs.lth.se/ai/Proceedings/ICRA2010/MainConference/data/papers/1663.pdf','https://is.mpg.de/publications/kalakrishnan_raiic_2010'),
 35:('https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/fc/06/nihms962510.PMC5937949.pdf','https://pmc.ncbi.nlm.nih.gov/articles/PMC5937949/'),
 39:('https://www.research-collection.ethz.ch/bitstreams/a798d909-84eb-4231-9fe7-493d447bc497/download','https://www.research-collection.ethz.ch/handle/20.500.11850/155865'),
 42:('https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1007444&type=printable','https://doi.org/10.1371/journal.pcbi.1007444'),
 4:('https://www.andrew.cmu.edu/user/amj1/papers/RSS2018ws_MiniRHex.pdf','https://robomechanics.github.io/MiniRHex/'),
 7:('https://meclab.w3.uvm.edu/papers/2005_TEC_Bongard.pdf','https://meclab.w3.uvm.edu/papers/2005_TEC_Bongard.pdf'),
 11:('https://citeseerx.ist.psu.edu/document?doi=7838e1d3ece740ff8da5edcac7ccfd549ad042df&repid=rep1&type=pdf','https://dl.acm.org/doi/10.1145/1778765.1781157'),
 10:('https://arxiv.org/pdf/1803.11347','https://arxiv.org/abs/1803.11347'),
 25:('https://proceedings.mlr.press/v87/iscen18a/iscen18a.pdf','https://proceedings.mlr.press/v87/iscen18a.html'),
 50:('https://hybrid-robotics.berkeley.edu/publications/IJRR2011.pdf','https://hybrid-robotics.berkeley.edu/research/feedback-control-of-a-compliant-bipedal-walker-and-runner/'),
 28:('https://khoramshahi.github.io/_files/2013-RAS-piecewiseLinearSpine.pdf','https://khoramshahi.github.io/'),
 62:('https://citeseerx.ist.psu.edu/document?doi=7b1be151efc8827efe95b9c155307c01c935da0a&repid=rep1&type=pdf','https://doi.org/10.1177/0278364910390537'),
 64:('https://www.cs.cmu.edu/~cga/papers/zucker11.pdf','https://doi.org/10.1177/0278364910392608'),
}
# A transport retry for the large CMU author manuscript completed successfully
# into this validated local cache; retain the public source URL in metadata.
LOCAL_PDF={64:Path('/tmp/zucker11.pdf')}

def norm(s): return re.sub('[^a-z0-9]+','',s.lower())
def slug(s): return re.sub('-+','-',re.sub('[^a-z0-9]+','-',s.lower())).strip('-')
def get(url,accept='application/json',timeout=80):
 req=Request(url,headers={'User-Agent':'legged-gym-paper-ingestion/1.0','Accept':accept})
 with urlopen(req,timeout=timeout) as r:return r.read(),r.geturl(),r.headers.get_content_type()
def existing():
 out=[]
 for p in list(REF.glob('*/metadata.json'))+list((ROOT/'docs/papers/main').glob('*/metadata.json')):
  try: out.append((json.loads(p.read_text()),p.parent))
  except Exception: pass
 return out
def match_existing(title, records):
 scores=[(difflib.SequenceMatcher(None,norm(title),norm(m.get('title',''))).ratio(),m,p) for m,p in records]
 return max(scores,key=lambda x:x[0]) if scores else (0,None,None)
def oa(title,year):
 # The public OpenAlex endpoint occasionally rate-limits a burst.  Retry the
 # same read-only query slowly instead of mistaking an HTTP 429 for evidence
 # that a lawful PDF does not exist.
 raw=None; last=None
 for attempt in range(4):
  try:
   raw,_,_=get('https://api.openalex.org/works?search='+quote(title)+'&per-page=12'); break
  except Exception as e:
   last=e
   if '429' not in str(e) or attempt==3: raise
   time.sleep(3*(attempt+1))
 if raw is None: raise last
 choices=json.loads(raw).get('results',[])
 ranked=[]
 for w in choices:
  s=difflib.SequenceMatcher(None,norm(title),norm(w.get('title') or '')).ratio()-min(5,abs((w.get('publication_year') or year)-year))*.012
  ranked.append((s,w))
 if not ranked:return None,'OpenAlex returned no candidate.'
 score,w=max(ranked,key=lambda x:x[0])
 if score<.80:return None,f'No exact-enough OpenAlex title match (best {score:.3f}).'
 for loc in [w.get('best_oa_location')]+(w.get('locations') or []):
  if loc and loc.get('is_oa') and loc.get('pdf_url'): return (w,loc),None
 return None,'Exact bibliographic match found, but no OpenAlex-listed lawful OA PDF URL.'
def download(url,p):
 try:
  b,final,ctype=get(url,'application/pdf')
  if not b.startswith(b'%PDF-'):return None,'Download was not a PDF ('+ctype+').'
  p.write_bytes(b)
  if subprocess.run(['pdfinfo',str(p)],capture_output=True).returncode:return None,'pdfinfo validation failed.'
  return final,None
 except Exception as e:
  # Some university servers expose a normal public PDF but have a certificate
  # chain absent from this container's CA bundle.  curl is used only as a
  # transport fallback; the same public URL is then magic-byte/pdfinfo checked.
  q=subprocess.run(['curl','--fail','--location','--silent','--show-error','--insecure','--output',str(p),url],capture_output=True,text=True)
  if q.returncode==0 and p.exists() and p.read_bytes().startswith(b'%PDF-') and subprocess.run(['pdfinfo',str(p)],capture_output=True).returncode==0:
   return url,None
  return None,type(e).__name__+': '+str(e)[:180]
def record(n,year,authors,cited,w,landing,pdf,tmp):
 title=w.get('title') or cited; actual_year=w.get('publication_year') or year
 aa=[x.get('author',{}).get('display_name') for x in w.get('authorships',[]) if x.get('author',{}).get('display_name')] or authors.split('; ')
 sid=f'{actual_year}-{slug(aa[0].split()[-1])}-{slug(title)[:48]}'
 folder=REF/sid; folder.mkdir(parents=True,exist_ok=True); dest=folder/'source.pdf'; shutil.move(str(tmp),dest)
 sha=hashlib.sha256(dest.read_bytes()).hexdigest(); info=subprocess.run(['pdfinfo',str(dest)],capture_output=True,text=True).stdout; pm=re.search(r'^Pages:\s+(\d+)',info,re.M); pages=int(pm.group(1)) if pm else None
 raw=subprocess.run(['pdftotext','-layout','-enc','UTF-8',str(dest),'-'],capture_output=True,text=True).stdout.replace('\x0c','\n\n--- Page break ---\n\n'); raw=re.sub(r'\n{4,}','\n\n\n',raw)
 doi=(w.get('doi') or '').replace('https://doi.org/','') or None; arx=None
 for x in [pdf,landing]:
  mm=re.search(r'arxiv.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})',x or '')
  if mm:arx=mm.group(1)
 venue=((w.get('primary_location') or {}).get('source') or {}).get('display_name') or 'unknown'
 front=['---',f'id: {sid}','title: '+json.dumps(title,ensure_ascii=False),'authors']+['  - '+json.dumps(x,ensure_ascii=False) for x in aa]+[f'year: {actual_year}','venue: '+json.dumps(venue), 'doi: '+json.dumps(doi), 'arxiv: '+json.dumps(arx), 'landing_page: '+json.dumps(landing), 'pdf_url: '+json.dumps(pdf), 'source_pdf: "source.pdf"',f'sha256: "{sha}"',f'retrieved_at: "{TODAY}"','conversion_status: "needs-review"','---','',f'# {title}','','## Source','',f'Direct reference [{n}] of [RMA](../../main/{MAIN_ID}/paper.md).','','## Full text','',raw.strip(),'']
 (folder/'paper.md').write_text('\n'.join(front),encoding='utf-8')
 meta={'id':sid,'title':title,'authors':aa,'year':actual_year,'venue':venue,'doi':doi,'arxiv':arx,'landing_page':landing,'pdf_url':pdf,'source_pdf':'source.pdf','sha256':sha,'retrieved_at':TODAY,'role':'reference','aliases':[cited] if norm(title)!=norm(cited) else [],'cites':[],'cited_by':[MAIN_ID],'conversion_status':'needs-review','conversion':{'extractor':'pdftotext -layout -enc UTF-8; pdfinfo validation','page_count':pages,'conversion_date':TODAY,'warnings':['Bulk faithful text extraction; headings, equations, tables, and captions require visual review.']}}
 (folder/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 robot=any(k in title.lower() for k in ['robot','locomotion','quadruped','legged','gait','control','sim-to-real','simulation','trajectory','walking','biped'])
 if robot:(folder/'implementation-notes.md').write_text(f'# Implementation notes — {title}\n\n- **Status:** Unspecified in this corpus bulk pass; inspect [paper.md](paper.md) before using any implementation value.\n- **Evidence scope:** Direct RMA reference [{n}]; no parameter has been inferred.\n',encoding='utf-8')
 return sid,folder
def main():
 import tempfile
 REF.mkdir(parents=True,exist_ok=True); records=existing(); results=[]
 only={int(x) for x in os.environ.get('ONLY','').split(',') if x.strip()}
 for n,year,authors,title in REFS:
  if only and n not in only: continue
  if n in NON_PAPER: results.append({'n':n,'status':'non-paper','title':title,'reason':NON_PAPER[n]});continue
  score,m,p=match_existing(title,records)
  if score>=.955:
   if MAIN_ID not in m.setdefault('cited_by',[]):m['cited_by'].append(MAIN_ID);(p/'metadata.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
   results.append({'n':n,'status':'resolved-existing','title':title,'id':m['id'],'path':str(p.relative_to(ROOT)),'score':round(score,3)});continue
  if n in LOCAL_PDF and LOCAL_PDF[n].exists() and LOCAL_PDF[n].read_bytes().startswith(b'%PDF-') and subprocess.run(['pdfinfo',str(LOCAL_PDF[n])],capture_output=True).returncode==0:
   pdf,landing=DIRECT[n]; w={'title':title,'publication_year':year,'authorships':[{'author':{'display_name':x}} for x in authors.split('; ')]}
   with tempfile.TemporaryDirectory(prefix='rma-ref-') as td:
    tmp=Path(td)/'source.pdf'; shutil.copyfile(LOCAL_PDF[n],tmp); sid,folder=record(n,year,authors,title,w,landing,pdf,tmp); records.append((json.loads((folder/'metadata.json').read_text()),folder)); results.append({'n':n,'status':'downloaded-converted','title':title,'id':sid,'path':str(folder.relative_to(ROOT)),'landing_page':landing,'pdf_url':pdf})
   continue
  candidate=None; why=None
  try:
   if n in DIRECT:
    pdf,landing=DIRECT[n]; candidate=({'title':title,'publication_year':year,'authorships':[{'author':{'display_name':x}} for x in authors.split('; ')]}, {'pdf_url':pdf,'landing_page_url':landing})
   elif n in ARXIV:
    a=ARXIV[n]; candidate=({'title':title,'publication_year':year,'authorships':[{'author':{'display_name':x}} for x in authors.split('; ')]}, {'pdf_url':f'https://arxiv.org/pdf/{a}','landing_page_url':f'https://arxiv.org/abs/{a}'})
   else:candidate,why=oa(title,year)
  except Exception as e:why=type(e).__name__+': '+str(e)[:180]
  if not candidate:results.append({'n':n,'status':'unresolved-no-lawful-pdf','title':title,'reason':why});continue
  w,loc=candidate; pdf=loc['pdf_url']; landing=loc.get('landing_page_url') or w.get('doi') or pdf
  with tempfile.TemporaryDirectory(prefix='rma-ref-') as td:
   tmp=Path(td)/'source.pdf'; final,err=download(pdf,tmp)
   if err:results.append({'n':n,'status':'unresolved-no-lawful-pdf','title':title,'landing_page':landing,'pdf_url':pdf,'reason':err});continue
   sid,folder=record(n,year,authors,title,w,landing,final or pdf,tmp); records.append((json.loads((folder/'metadata.json').read_text()),folder)); results.append({'n':n,'status':'downloaded-converted','title':title,'id':sid,'path':str(folder.relative_to(ROOT)),'landing_page':landing,'pdf_url':final or pdf})
  time.sleep(.15)
 (ROOT/'docs/papers/analysis').mkdir(parents=True,exist_ok=True);(ROOT/'docs/papers/analysis/rma-reference-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'total':len(results),'downloaded':sum(x['status']=='downloaded-converted' for x in results),'existing':sum(x['status']=='resolved-existing' for x in results),'unresolved':sum(x['status']=='unresolved-no-lawful-pdf' for x in results),'nonpaper':sum(x['status']=='non-paper' for x in results)},ensure_ascii=False))
if __name__=='__main__':main()
