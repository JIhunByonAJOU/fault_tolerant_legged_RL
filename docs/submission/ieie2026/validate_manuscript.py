"""Audit manuscript numbers, references, PDF layout logs and anonymization."""
import argparse,csv,hashlib,json,re,shutil,subprocess,zlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
LATEX=BASE/'private/latex'
DATA=ROOT/'datasets/ieie2026_fixed2112'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(*args):return subprocess.check_output(args,text=True,encoding='utf-8')
def brace_groups(text):
 result=[];depth=0;start=None
 for i,c in enumerate(text):
  if c=='{':
   if depth==0:start=i+1
   depth+=1
  elif c=='}':
   depth-=1
   if depth==0:result.append(text[start:i])
 assert depth==0,text
 return result

def audit_pdf_references(pdf,aux,body,bibs):
 import xml.etree.ElementTree as ET
 contents=run('pdftotext','-layout',str(pdf),'-')
 assert not re.search(r'\?\?|\[\s*\?\s*\]',contents),str(pdf)+' has unresolved references'
 labels={}
 for line in aux.splitlines():
  match=re.match(r'\\newlabel\{([^}]+)\}(\{.*\})$',line)
  if match:
   fields=brace_groups(brace_groups(match[2])[0]);labels[match[1]]=fields
 source_labels=re.findall(r'\\label\{([^}]+)\}',body)
 assert len(source_labels)==len(set(source_labels)),'Duplicate source labels'
 refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',body)
 assert set(refs)<=set(labels),(set(refs)-set(labels))
 destinations={m[2]:int(m[1]) for m in re.finditer(r'^\s*(\d+)\s+.*"([^"]+)"\s*$',run('pdfinfo','-dests',str(pdf)),re.M)}
 expected={}
 for label in source_labels:
  number,page,_,dest,*rest=labels[label]
  assert destinations.get(dest)==int(page),(label,dest,page)
  expected[({'fig':'그림','tab':'표','eq':'식'}[label.split(':')[0]],int(number))]=int(page)
 bib_pages={}
 for i,key in enumerate(bibs,1):
  assert re.search(r'\\bibcite\{'+re.escape(key)+r'\}\{'+str(i)+r'\}',aux),(key,i)
  assert 'cite.'+key in destinations,key
  bib_pages[i]=destinations['cite.'+key]
 tree=ET.fromstring(run('pdftohtml','-q','-xml','-hidden','-stdout',str(pdf)))
 seen=set();cited=set();links=0
 prefix='';category=None;previous_top=None;previous_right=None;citation=None
 for text in tree.iter('text'):
  top=float(text.get('top',0));left=float(text.get('left',0))
  if previous_top is None or abs(top-previous_top)>3 or left<previous_right-3 or left-previous_right>35:
   prefix='';category=None
  prefix+=text.text or ''
  previous_top=top;previous_right=left+float(text.get('width',0))
  for child in text:
   value=''.join(child.itertext())
   if child.tag=='a':
    target=re.search(r'#(\d+)$',child.get('href',''));assert target is not None,child.attrib
    page=int(target[1]);links+=1
    marker=re.search(r'(그림|표|식)\s*$',prefix)
    if marker:category=marker[1]
    # Superscript [2, 5] may be split into several linked font runs by pdftohtml.
    # Retain each numeral's destination while accumulating the complete bracket group.
    is_citation=citation is not None or '[' in value
    for char in value:
     if char=='[':citation=[]
     elif citation is not None:
      if char==']':
       group=''.join(c for c,_ in citation)
       assert re.fullmatch(r'[\d,\s\-]+',group),group
       for match in re.finditer(r'\d+',group):
        number=int(match[0]);targets={citation[i][1] for i in range(match.start(),match.end())}
        assert targets=={bib_pages[number]},(number,targets,bib_pages[number]);cited.add(number)
       citation=None
      else:citation.append((char,page))
    if not is_citation and category:
     for number in map(int,re.findall(r'\d+',value)):
      key=(category,number);assert expected[key]==page,(key,page,expected[key]);seen.add(key)
   prefix+=value+(child.tail or '')
 assert citation is None,'Unclosed linked citation group'
 assert cited==set(bib_pages),('Citations without verified clickable links',set(bib_pages)-cited)
 assert seen==set(expected),('References without verified clickable links',set(expected)-seen)
 return {'cross_reference_labels':len(source_labels),'reference_occurrences':len(refs),'bibliography_entries':len(bibs),'internal_pdf_links':links,'all_reference_target_pages_verified':True,'all_citation_target_pages_verified':True,'no_duplicate_labels':True,'no_unresolved_reference_or_citation_markers':True,'reference_destinations':destinations}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--publish-previews',action='store_true');parser.add_argument('--require-page-limits',action='store_true');parser.add_argument('--build-dir',type=Path,default=LATEX/'build');args=parser.parse_args();build=args.build_dir
 configs=[ROOT/'.vscode/settings.json',ROOT/'ieie-paper.code-workspace',LATEX/'.vscode/settings.json',LATEX/'ieie-paper.code-workspace']
 for config in configs:
  settings=json.loads(config.read_text(encoding='utf-8'));settings=settings.get('settings',settings)
  assert settings['latex-workshop.latex.recipe.default']=='LuaLaTeX (한글 논문)',str(config)
  assert settings['latex-workshop.latex.autoBuild.run']=='onSave',str(config)
  for tool in settings['latex-workshop.latex.tools']:
   assert '-lualatex' in tool['args'] and '-xelatex' not in tool['args'],str(config)
 for entry in ['proceedings.tex','main.tex']:assert '\\RequireLuaTeX' in (LATEX/entry).read_text(encoding='utf-8')
 records=json.loads((DATA/'records.json').read_text(encoding='utf-8'));assert len(records)==26352
 evidence={}
 for policy in ['Normal','Student']:
  rs=[r for r in records if r['policy']==policy and r['condition']['loss']==1]
  assert len(rs)==2592
  evidence[policy]={'n':len(rs),'survival_percent':100*sum(r['survived'] for r in rs)/len(rs),'goal_percent':100*sum(r['termination']=='goal' for r in rs)/len(rs)}
  for key in ['command_vx_rmse','command_vy_rmse','command_wz_rmse','cross_track_rmse']:evidence[policy][key]=sum(r[key] for r in rs)/len(rs)
 for pol,alive,goal in [('Normal',1350,618),('Student',2313,2248)]:
  assert abs(evidence[pol]['survival_percent']-100*alive/2592)<1e-12
  assert abs(evidence[pol]['goal_percent']-100*goal/2592)<1e-12
 joint_case={}
 for policy in ['Normal','Student']:
  rs=[r for r in records if r['policy']==policy and r['condition']['loss']==1 and r['condition']['joint']==8]
  assert len(rs)==216
  counts=(sum(r['survived'] for r in rs),sum(r['termination']=='goal' for r in rs))
  assert counts==({'Normal':(188,3),'Student':(198,195)}[policy])
  joint_case[policy]={'n':len(rs),'survival_percent':100*counts[0]/len(rs),'completion_percent':100*counts[1]/len(rs)}
 for joint,fallen in [(1,46),(3,6),(4,61)]:
  rs=[r for r in records if r['policy']=='Student' and r['condition']['loss']==1 and r['condition']['joint']==joint]
  assert len(rs)==216 and sum(not r['survived'] for r in rs)==fallen
 table=list(csv.DictReader((BASE/'paper_assets/task_metrics.csv').open()))
 metrics=['command_vx_rmse_trial_mean','command_vy_rmse_trial_mean','command_wz_rmse_trial_mean','cross_track_rmse_trial_mean']
 for family in ['forward','diagonal','lateral','sine']:
  n=next(r for r in table if r['policy']=='Normal' and r['family']==family and float(r['loss'])==1);s=next(r for r in table if r['policy']=='Student' and r['family']==family and float(r['loss'])==1)
  assert float(s['survival'])>float(n['survival'])
  for metric in metrics:assert float(s[metric])<float(n[metric])
 mobility=list(csv.DictReader((BASE/'paper_assets/task_mobility.csv').open()))
 for cell in mobility:
  rs=[r for r in records if r['policy']==cell['policy'] and r['profile']['family']==cell['family'] and r['condition']['loss']==1]
  assert len(rs)==int(cell['episodes'])
  assert sum(r['termination']=='goal' for r in rs)==int(cell['completed'])
  assert sum(r['survived'] for r in rs)==int(cell['survived'])
 pdfs={}
 references=(LATEX/'references.tex').read_text(encoding='utf-8');bibs=re.findall(r'\\bibitem\{([^}]+)\}',references)
 for name,limit in [('proceedings',5),('main',4)]:
  pdf=build/f'{name}.pdf';info=run('pdfinfo',str(pdf));pages=int(re.search(r'Pages:\s+(\d+)',info)[1])
  text=run('pdftotext','-layout',str(pdf),'-');log=(build/f'{name}.log').read_text(encoding='utf-8')
  assert log.startswith('This is LuaHBTeX') or log.startswith('This is LuaTeX'),name+' used the wrong TeX engine'
  assert not any(x in log for x in ['Overfull \\hbox','Overfull \\vbox','Missing character:','undefined references','Undefined control sequence','Fatal error','There were undefined references','Rerun to get cross-references right','Citation `'])
  assert '??' not in text
  assert '52.08%' in text and '89.24%' in text and '86.73%' in text
  body=(LATEX/('body.tex' if name=='proceedings' else 'body_review.tex')).read_text(encoding='utf-8')
  intro=body.split('\\section{서론}',1)[1].split('\\section{시스템 모델}',1)[0]
  assert not any(w in intro for w in ['RMA','ADAPT','기여는'])
  for token in ['fig:environment','tab:system','tab:observation','tab:reward','paper_workflow_clean.pdf','50\\times48','235+8=243','187','45','12']:
   assert token in body,(name,token)
  assert all(x not in body for x in ['task_metrics','\\url{','\\href{','계열로','·','base 링크','토크 저하 수준','손실률','보조 사례','보조 해석','고장 후 시간 상한','고장 배율','\\eta_j','관절속도','관절의 특성을 추정'])
  for caption in re.findall(r'\\caption(?:\[[^\]]*\])?\{([^\n]+)',body):
   assert not any(word in caption for word in ['음영','각 셀','상단','하단','스케일','이다.']),caption
  assert '발 디딤 위치와 다리의 움직임을 조절하여' in intro
  assert '교사, 학생 공동학습으로 고장 정보를 직접 입력받지 않는 관측 이력 기반 정책을 학습한다' in intro
  assert '이를 통해' in intro and '보인다.' in intro
  assert 'paper_survival_completion.pdf' in body and 'paper_tracking_rmse_2x2.pdf' in body
  aux=(build/f'{name}.aux').read_text(encoding='utf-8')
  for label,index in [('fig:survival',4),('fig:tracking',5),('fig:joint',6)]:assert ('\\newlabel{'+label+'}{{'+str(index)+'}') in aux,(name,label)
  assert all(x not in body for x in ['eq:survival','eq:rmse','eq:fault','tab:tasks','tab:encoders','tab:evaluation','음영','관절각'])
  assert len(re.findall(r'\\label\{eq:',body))==6
  assert '시행별 root mean square error (RMSE)로 계산한 뒤 동일 가중치로 평균' in body
  reference_audit=audit_pdf_references(pdf,aux,body,bibs)
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
  assert not re.search(r'\d',conclusion),'Conclusion must describe results without numerical repetition'
  assert 'm_j(1-d_j)' in body and '$d_j$는 관절별 토크 저하율' in body
  assert 'd_j\\in[0,1]' not in body and r'd_j\in\{0.0,0.2,0.4,0.6,0.8,1.0\}' in body
  assert '교사 인코더와 같은 은닉층 크기의 MLP' not in body
  architecture=body.split('\\subsection{관측 및 이력 표현}',1)[1].split('\\subsection{교사, 학생 공동학습}',1)[0]
  assert all(x in architecture for x in ['fig:architecture','multi-layer perceptron (MLP)','convolutional neural network (CNN)','[512, 256, 128]'])
  training=body.split('\\subsection{정책 학습 설정과 과정}',1)[1].split('\\section{시뮬레이션 결과',1)[0]
  assert '마찰과 하중, 모터 강도가 달라져도' in training
  assert 'multi-layer perceptron' not in training
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
  pdfs[name]={'reference_audit':reference_audit,'pages':pages,'target_pages':limit,'page_limit_met':pages<=limit,'photos_three_tables_and_dimensions_verified':True,'sha256':digest(pdf),'all_pdf_fonts_embedded':True,'no_overfull_or_missing_glyphs':True,'citations_sentence_end_attached_and_first_appearance_order':True,'all_figures_tables_equations_referenced':True,'conclusion_has_no_stagnation_claim':True,'anonymous_text_and_author_link_removed':True if name=='main' else None}
 a=json.loads((DATA/'figures/style_audit.json').read_text(encoding='utf-8'));assert a['font_size_pt']==9
 assert a['heatmap']['color_limits_percent']==[.4,30]
 assert a['heatmap']['normalization']=='logarithmic' and not a['heatmap']['values_log_transformed']
 assert a['compact_performance']['data_identical_to_individual_curves']
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
 cells=list(csv.DictReader((DATA/'figures/paper_student_fall_rate_heatmap.csv').open(encoding='utf-8-sig')))
 assert len(cells)==60
 for cell in cells:
  rs=[r for r in records if r['policy']=='Student' and r['condition']['joint']==int(cell['joint_index']) and r['condition']['loss']==float(cell['degradation'])]
  assert len(rs)==216 and sum(r['survived'] for r in rs)==int(cell['survived'])
  assert int(cell['fallen'])==216-int(cell['survived'])
  assert abs(float(cell['fall_rate_percent'])-100*int(cell['fallen'])/216)<1e-12
  assert abs(float(cell['fall_rate_percent'])+float(cell['survival_percent'])-100)<1e-12
 paired_audit=json.loads((BASE/'paper_assets/survival_completion_audit.json').read_text(encoding='utf-8'))
 assert paired_audit['records_sha256']==digest(DATA/'records.json')
 assert paired_audit['font_family']=='Times New Roman' and paired_audit['font_size_pt']==9
 assert paired_audit['width_mm']==80 and paired_audit['height_mm']==44
 assert not paired_audit['internal_title'] and not paired_audit['outside_text'] and not paired_audit['text_overlaps']
 paired_rows=list(csv.DictReader((BASE/'paper_assets/survival_completion.csv').open()))
 assert len(paired_rows)==12 and len(paired_audit['values'])==12
 for row in paired_rows:
  rs=[r for r in records if r['policy']==row['policy'] and r['condition']['loss']==float(row['loss'])]
  survived=sum(r['survived'] for r in rs);completed=sum(r['termination']=='goal' for r in rs)
  assert len(rs)==int(row['episodes']) and survived==int(row['survived']) and completed==int(row['completed'])
  assert abs(float(row['survival_percent'])-100*survived/len(rs))<1e-12
  assert abs(float(row['completion_percent'])-100*completed/len(rs))<1e-12
  plotted=next(p for p in paired_audit['values'] if p['policy']==row['policy'] and p['loss']==float(row['loss']))
  assert plotted['survival_percent']==float(row['survival_percent']) and plotted['completion_percent']==float(row['completion_percent'])
  for metric,flag in [('survival','survived'),('completion','termination')]:
   seed_rates=[]
   for seed in [1101,1102,1103]:
    group=[r for r in rs if r['seed']==seed]
    successes=sum(r['survived'] if flag=='survived' else r['termination']=='goal' for r in group)
    seed_rates.append(100*successes/len(group))
   assert abs(plotted[metric+'_seed_min']-min(seed_rates))<1e-12
   assert abs(plotted[metric+'_seed_max']-max(seed_rates))<1e-12
  assert 0<=completed<=survived<=len(rs)
 asset_audit=json.loads((BASE/'paper_assets/figure_audit.json').read_text(encoding='utf-8'))
 for name,fig in asset_audit.items():assert fig['font_size_pt']==9 and not fig['internal_title'] and not fig['outside_text'],name
 abstract=(LATEX/'abstract-proceedings.tex').read_text(encoding='utf-8')
 assert abstract.splitlines()[1].startswith('We propose a reinforcement learning locomotion policy')
 assert 'in Isaac Gym using a Unitree A1 quadruped model' in abstract
 assert 'one seed each' in abstract
 assert 84<=len(abstract.split('\n',1)[1].split())<=94
 assert f"{evidence['Student']['survival_percent']-evidence['Normal']['survival_percent']:.1f} percentage points" in abstract
 cross_gain=100*(1-evidence['Student']['cross_track_rmse']/evidence['Normal']['cross_track_rmse'])
 assert f'{cross_gain:.1f}\\%' in abstract
 assert 'pre-termination cross-track' in abstract and all(x not in abstract for x in ['0.226','0.120','0.262','0.112','0.636','0.206'])
 architecture_text=run('pdftotext',str(BASE/'private/assets/paper_workflow_clean.pdf'),'-')
 assert not any(x in architecture_text for x in ['Training:', 'Critic also receives', 'Inference:'])
 assert all(x in architecture_text for x in ['Teacher MLP','Student CNN','Feature mixing','Shared Actor','235 + 8 = 243','output: 12D'])
 assert '시행별 root mean square error' in (LATEX/'frontmatter.tex').read_text(encoding='utf-8')
 result={'editorial_revision':'20261018_compact_abstract_introduction_and_expanded_conclusion','user_introduction_applied':True,'paper_has_no_data_file_paths_or_repository_instructions':True,'task_survival_and_completion_counts_verified':True,'restoration_base_commit':'0d34639','submission_ready':False,'page_limits_met':all(x['page_limit_met'] for x in pdfs.values()),'heatmap_metric':'fall_rate_percent = 100 - survival_percent','heatmap_survival_complements_verified':True,'heatmap_color_limits_percent':[.4,30],'heatmap_color_normalization':'logarithmic, original rates unchanged','active_legacy_architecture':'private/assets/paper_workflow_clean.pdf; three peripheral notes removed; original boxes, arrows, dimensions and font styling retained','records_sha256':digest(DATA/'records.json'),'primary_d100_evidence':evidence,'per_family_d100_all_primary_metrics_improved':True,'pdfs':pdfs,'new_quantitative_figure_font_size_pt':9,'new_quantitative_figure_titles_absent':True,'all_plotted_points_csv_max_difference':max(differences),'all_60_heatmap_counts_verified':True,'generated_assets_audit_including_unused_simplified_architecture':asset_audit,'heatmap_cells':a['heatmap']['cells'],'heatmap_minimum_text_contrast':a['heatmap']['minimum_text_background_contrast'],'raw_simulation_reexecuted':False,'training_reexecuted':False}
 for name,details in pdfs.items():
  source=build/f'{name}.pdf';preview=LATEX/'preview'/f'{name}.pdf'
  if args.publish_previews:
   assert build.resolve()==(LATEX/'build').resolve(),'Publish the canonical editor build only'
   shutil.copyfile(source,preview)
  assert digest(preview)==details['sha256'],name+' preview differs from validated build'
  details['editor_build_and_preview_byte_identical']=True
 result['rl_calf_full_loss_linked_metric_case']=joint_case
 result['survival_and_completion_panel']=paired_audit
 result['metric_definitions_in_prose_without_equations_8_and_9']=True
 result['task_table_replaced_by_completion_curve']=True
 result['encoder_table_removed_with_dimensions_retained']=True
 result['evaluation_table_replaced_by_condition_prose']=True
 result['manuscript_has_no_uncertainty_band_prose']=True
 result['all_four_editor_configurations_use_lualatex_on_save']=True
 result['wrong_engine_guard_in_both_entry_files']=True
 out=BASE/'paper_assets/manuscript_validation.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
 if args.require_page_limits and not result['page_limits_met']:raise SystemExit('Page limits not met; restored content retained.')
if __name__=='__main__':main()
