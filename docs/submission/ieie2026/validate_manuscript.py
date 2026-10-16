"""Audit manuscript numbers, references, PDF layout logs and anonymization."""
import csv,hashlib,json,re,subprocess,zlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
LATEX=BASE/'private/latex'
DATA=ROOT/'datasets/ieie2026_fixed2112'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(*args):return subprocess.check_output(args,text=True)
def main():
 records=json.loads((DATA/'records.json').read_text());assert len(records)==26352
 evidence={}
 for policy in ['Normal','Student']:
  rs=[r for r in records if r['policy']==policy and r['condition']['loss']==1]
  assert len(rs)==2592
  evidence[policy]={'n':len(rs),'survival_percent':100*sum(r['survived'] for r in rs)/len(rs),'goal_percent':100*sum(r['termination']=='goal' for r in rs)/len(rs)}
  for key in ['command_vx_rmse','command_vy_rmse','command_wz_rmse','cross_track_rmse']:evidence[policy][key]=sum(r[key] for r in rs)/len(rs)
 for pol,alive,goal in [('Normal',1350,618),('Student',2313,2248)]:
  assert abs(evidence[pol]['survival_percent']-100*alive/2592)<1e-12
  assert abs(evidence[pol]['goal_percent']-100*goal/2592)<1e-12
 table=list(csv.DictReader((BASE/'paper_assets/task_metrics.csv').open()))
 metrics=['command_vx_rmse_trial_mean','command_vy_rmse_trial_mean','command_wz_rmse_trial_mean','cross_track_rmse_trial_mean']
 for family in ['forward','diagonal','lateral','sine']:
  n=next(r for r in table if r['policy']=='Normal' and r['family']==family and float(r['loss'])==1);s=next(r for r in table if r['policy']=='Student' and r['family']==family and float(r['loss'])==1)
  assert float(s['survival'])>float(n['survival'])
  for metric in metrics:assert float(s[metric])<float(n[metric])
 pdfs={}
 references=(LATEX/'references.tex').read_text();bibs=re.findall(r'\\bibitem\{([^}]+)\}',references)
 for name,limit in [('proceedings',5),('main',4)]:
  pdf=LATEX/'build'/f'{name}.pdf';info=run('pdfinfo',str(pdf));pages=int(re.search(r'Pages:\s+(\d+)',info)[1]);assert pages<=limit,(name,pages)
  text=run('pdftotext','-layout',str(pdf),'-');log=(LATEX/'build'/f'{name}.log').read_text()
  assert not any(x in log for x in ['Overfull \\hbox','Overfull \\vbox','Missing character:','undefined references','Citation `'])
  assert '52.08%' in text and '89.24%' in text and '86.73%' in text
  body=(LATEX/('body.tex' if name=='proceedings' else 'body_review.tex')).read_text()
  citations=re.findall(r'\\cite\{([^}]+)\}',body);order=[]
  for keys in citations:
   for key in keys.split(','):
    if key not in order:order.append(key)
  assert order==bibs,(name,order,bibs)
  assert not re.search(r'\s+\\cite\{',body)
  assert not re.search(r'\\cite\{[^}]+\}(?!\.)',body)
  labels=re.findall(r'\\label\{([^}]+)\}',body)
  for label in labels:assert re.search(r'\\(?:eqref|ref)\{'+re.escape(label)+r'\}',body),label
  assert not any('\u3040'<=c<='\u30ff' or '\u4e00'<=c<='\u9fff' for c in body)
  conclusion=body.split('\\section{결론}',1)[1].split('\\input{references.tex}',1)[0]
  assert not any(w in conclusion for w in ['정체','버티기','Q_{','전방 순변위'])
  if name=='main':
   assert not any(w.lower() in text.lower() for w in ['Ajou','Jihun','Soyi','변지훈','정소이','ajou.ac.kr'])
   assert '\\href' not in body
  if name=='main':
   raw=pdf.read_bytes();objects=[raw]
   for match in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream',raw,re.S):
    try:objects.append(zlib.decompress(match[1]))
    except zlib.error:pass
   joined=b'\n'.join(objects).lower()
   for token in [b'/uri',b'jihun',b'soyi',b'ajou',b'josephx03021',b'/home/']:assert token not in joined,token
   for field in ['Author','Title','Subject','Keywords']:
    found=re.search(r'^'+field+r':(.*)$',info,re.M)
    assert found is None or not found[1].strip(),(field,found[1])
  fonts=run('pdffonts',str(pdf)).splitlines()[2:]
  assert fonts
  for line in fonts:assert line.split()[-5]=='yes',line
  pdfs[name]={'pages':pages,'sha256':digest(pdf),'all_pdf_fonts_embedded':True,'no_overfull_or_missing_glyphs':True,'citations_sentence_end_attached_and_first_appearance_order':True,'all_figures_tables_equations_referenced':True,'conclusion_has_no_stagnation_claim':True,'anonymous_text_and_author_link_removed':True if name=='main' else None}
 a=json.loads((DATA/'figures/style_audit.json').read_text());assert a['font_size_pt']==9
 for name,f in a['figures'].items():assert not f['internal_title'] and not f['outside_text'] and f['text_sizes_pt']==[9.0],name
 for f in list((BASE/'paper_assets').glob('paper_*.pdf')):
  for line in run('pdffonts',str(f)).splitlines()[2:]:assert 'TimesNewRoman' in line and line.split()[-5]=='yes',line
 # Validate every plotted point and every heatmap denominator against independent tables/records.
 aggregate=list(csv.DictReader((DATA/'reports/by_loss_with_pooled_rmse.csv').open()))
 differences=[]
 for metric,policies in a['curve_values'].items():
  for policy,series in policies.items():
   for rate,value in zip(series['degradation'],series['mean']):
    row=next(r for r in aggregate if r['policy']==policy and float(r['loss'])==rate)
    expected=100*float(row['survival']) if metric=='survived' else float(row[metric+'_trial_mean'])
    differences.append(abs(value-expected))
 assert max(differences)<1e-12
 cells=list(csv.DictReader((DATA/'figures/paper_student_survival_heatmap.csv').open(encoding='utf-8-sig')))
 assert len(cells)==60
 for cell in cells:
  rs=[r for r in records if r['policy']=='Student' and r['condition']['joint']==int(cell['joint_index']) and r['condition']['loss']==float(cell['degradation'])]
  assert len(rs)==216 and sum(r['survived'] for r in rs)==int(cell['survived'])
 asset_audit=json.loads((BASE/'paper_assets/figure_audit.json').read_text())
 for name,fig in asset_audit.items():assert fig['font_size_pt']==9 and not fig['internal_title'] and not fig['outside_text'],name
 result={'records_sha256':digest(DATA/'records.json'),'primary_d100_evidence':evidence,'per_family_d100_all_primary_metrics_improved':True,'pdfs':pdfs,'active_figure_font_size_pt':9,'active_figure_titles_absent':True,'all_plotted_points_csv_max_difference':max(differences),'all_60_heatmap_counts_verified':True,'asset_figure_audit':asset_audit,'heatmap_cells':a['heatmap']['cells'],'heatmap_minimum_text_contrast':a['heatmap']['minimum_text_background_contrast'],'raw_simulation_reexecuted':False,'training_reexecuted':False}
 out=BASE/'paper_assets/manuscript_validation.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
