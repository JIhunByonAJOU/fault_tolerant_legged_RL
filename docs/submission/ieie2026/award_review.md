# 이전 경진대회·수상 사례 조사

확인일: 2026-10-02. **수상 확인**, **발표 프로그램 확인**, **논문 전문 확보**를 구분했다. 수상 보도에서 전문을 읽었다고 주장하거나 제목만으로 수상 이유를 단정하지 않는다.

## 대회 구분

사용자가 언급한 '2026 춘계'와 동일한 대한전자공학회 학부생 논문경진대회는 이번 공식 사이트 검색에서 확인되지 않았다. 대신 2026-06-24 **하계 AISP 인공지능 학부생 논문경진대회**가 확인된다. 추계 학부생 대회와 하계 AI 소사이어티 대회는 별도 프로그램/규정이다. [2026 하계 공식 사이트](https://bpai2026.github.io/), [2025 하계 공식 사이트](https://bpai2025.github.io/), [2024 추계 공식 안내](https://conf2024f.ieieweb.org/2024f/pages/undergraduate_competition.vm).

## 원자료로 확인된 사례

| 연도/대회 | 연구/저자 | 확인 범위 | 우리에게 주는 참고점 |
|---|---|---|---|
| 2024 추계 학부생 | DRAM기반 PIM을 활용한 벡터 유사도 검색 가속 / 노해찬누리·정민, 이화여대 | **우수상** 대학 공식보도. 전문 미확보 | 기존 RAG 검색 병목을 특정하고 Brute-force/HNSW·거리연산을 실제 구현 후 평가. '기존기술 응용'도 좁은 문제와 성능 검증을 갖추면 연구 구성이 가능 |
| 2025 하계 AI | Wan2.1 기반 LoRA Fine-Tuning 기반 미술작품 동적 비주얼 생성 / 김동준, 한국기술교육대 | **우수상** 대학 공식보도. 전문 미확보 | 기존모델 이름보다 적용문제·수정방법·사용가능성을 연결. 보도는 기술 완성도/응용가능성을 언급하지만 정량지표 수치는 제공하지 않음 |
| 2026 하계 AI | 안전한 제지 건조 공정 제어를 위한 비용 인지 PPO와 배포 시점 행동 스케일링 / 유지오·이서영 등, 국립한밭대 | **장려상** 대학 공식보도+대회 프로그램. 전문 미확보 | PPO의 제어 안전성과 에너지효율을 목적별로 분리. 우리의 생존/기동/추종 분리와 가장 가까운 평가 논리 사례 |
| 2025 추계 | 경량 블록 암호 LEA-128의 FPGA 구현 / 김기훈·김재현·곽승현, 금오공대 | **우수논문상** 대학 공식보도. 학부생경진대회 상인지 보도만으로 정확한 부문 미확정. 전문 미확보 | 저전력·경량 응용목적과 효율적 구현을 연결. '추계 우수논문상'을 곧바로 '학부생경진대회 우수상'이라 부르지 않음 |

원출처:
- [이화여대 2024 수상 보도](https://myr.ewha.ac.kr/ewhaeleca/information/news.do?articleNo=738201&mode=view)
- [한국기술교육대 2025 수상 보도](https://www.koreatech.ac.kr/gallery.es?act=view&bid=0018&list_no=52690&mid=a10701020000)
- [한밭대 2026 PPO 수상 보도](https://hanbat.ac.kr/prog/bbsArticle/BBSMSTR_000000000055/view.do?mno=sub07_0404&nttId=B000000167663Na0fQ0c)
- [금오공대 2025 추계 수상 보도](https://cam.kumoh.ac.kr/ko/sub01_05_02.do?articleNo=547507&mode=view)

## 강화학습·로봇 관련 프로그램 사례

2025 하계 AI 후보 발표에는 `Gate-Level Cell Selection via Reinforcement Learning for Netlist cost Minimization`, `Conditional Flow Policy: Enhancing Robot Imitation Learning via Flow Matching Beyond Diffusion Policy`, 시각장애인 충돌회피 보조, 객체검출·가상벽 기반 자율주행 경로계획 등이 포함됐다. **프로그램 등재 사실만 확인**했고 각각의 수상 등급이나 논문 전문은 확보하지 않았다. 2025 사이트는 후보10편과 최우수1/우수3/장려6 시상 계획을 적지만 개별 순위표를 제공하지 않으므로 제목별 상을 추측하지 않는다. [2025 공식 프로그램](https://bpai2025.github.io/).

2026 하계 AI 프로그램에는 위 제지공정 PPO 및 IR-UWB 통신의 DQN 변조선택 연구가 있다. PPO 연구의 장려상은 위 대학 원자료로 따로 검증했다. [2026 공식 프로그램](https://bpai2026.github.io/).

## 논문 설계에 반영할 판단

다음은 수상 통계로 입증한 법칙이 아니라 확보한 사례를 바탕으로 한 **작성 전략**이다.

1. 문제를 좁힌다: '고장에 강한 4족로봇'보다 '단일 관절 출력저하 후 쓰러짐과 기동 상실을 분리하여 평가하는 관측이력 기반 정책'이 검증 가능하다.
2. 비교대상의 역할을 설명한다: 정상정책과 최종 학생의 실용 비교, 같은 최종 Actor의 특징 경로 교체, 이력 입력 개입을 구분한다. 특징 교체를 성능 상한으로 보거나 추론 개입을 재학습한 ablation으로 부르지 않는다. 과거 체크포인트 번호만 늘어놓는 비교는 기여 증거가 약하다.
3. 결과표의 숫자를 동작 의미로 번역한다: 0.5m/s 명령에서 20초간 목표10m 중 실제전진 거리·생존시간·정지시간을 함께 보여준다.
4. 지표마다 조건과 분모를 적는다: alive-only RMSE는 일찍 넘어진 정책의 오차가 작아 보일 수 있으므로 생존율과 항상 함께 보고한다.
5. 기존의 RMA/ADAPT/2026 latent-alignment 연구와 겹치는 부분을 인정한다. 자체 구현·저학습률·긴학습만을 새로운 알고리즘이라고 쓰지 않는다.
6. 4페이지 안에서 방법 그림1, 실험조건표1, 핵심 결과표1, 관절열화 히트맵/시간응답1~2로 질문-증거를 연결한다.

## 조사 한계

2024~2026 모든 수상 논문의 전문을 확보한 상태가 아니다. 대학 보도는 수상과 개괄을 입증하지만 baseline/ablation/통계/본문 구성 비교를 뒷받침하지 못한다. 2025·2026 하계 프로그램의 논문 공개 기간/논문집 접근 경로도 다르므로, 논문집 전체를 읽은 것처럼 인용하지 않는다. 현재 확보한 공식 사이트/대학 원자료를 먼저 활용하고 후속 전문 확보가 되면 이 문서에 별도로 표시한다.

## 수상 원문 추가 심층 조사 (2026-10-02)

보도·프로그램과 논문 자체를 별도로 추적했다. 수상사유의 심사평은 공개자료에서 확인되지 않는다. 아래 공개 전문을 분석한 내용도 '그 때문에 수상했다'는 인과적 설명으로 사용하지 않는다.

| 대상 | 추가 확보한 일차 논문 기록 | 전문 접근 및 해석 범위 |
|---|---|---|
|2024추계 우수상, 이화 노해찬누리·정민 PIM|[DBpia 원논문](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12036194), 정확한 제목/6저자/2024.11/pp185–189 **5p** 확인|수상은 이화 공식보도와 연결된다. 초록 공개, 로그인·기관/개인이용 경로가 있는 원문뷰어이나 무로그인 공개 PDF 미확보. 페이지별구성·ablation·seed는 분석하지 못함|
|2024추계 장려상, 경북 서보건 AR 자세추정|[경북대 공식소식](https://gsee.knu.ac.kr/content/news.html?fidx=103428&gtid=bodo&opt=&page=4&pg=vv&sword=)에서 정확제목 'AR 적용을 위한 딥러닝 기반 단안 RGB 객체의 자세 추정 및 CAD 모델 채색화', 1저자·지도교수 확인|보도 첨부는 사진, 논문첨부 아님. 제목/저자 검색에서 공개 전문 미확보; 로보틱스연관비전 사례이나 보행RL논문은 아님|
|2025하계 AISP, 로봇Conditional Flow Policy|[DBpia 원논문](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12332007), Chanhyuk Jung·Byoung Chul Ko,2025.6 pp2634–2637 **4p** 확인|공식프로그램 본선10편 포함 확인, 개별상등급은 미확인. 무로그인 전문 미확보. 아래 같은저자의 **다른 ICCVW논문**은 공개전문 확보|
|2025하계 AISP, RLCellSelection 강지원|공식프로그램 exacttitle 'Gate-Level Cell Selection via Reinforcement Learning for Netlist cost Minimization'|본문·학술메타데이터·개별등급 추가확보 못함. 제목만으로 state/action/reward/실험추정금지|
|2026하계 AISP 장려상, 한밭 비용인지PPO|한밭 공식수상보도+공식경진프로그램; [하계 전체프로그램PDF](https://conf.theieie.org/2026s/download/session_program_20260602.pdf) 인쇄p123, P13-21/GEP-0903로 동일제목6저자 일반포스터 기록도 확인|동일제목 일반발표와 경진대회 수상 기록을 구분. 제목검색/연구자검색/경진repo에서 무로그인전문 미확보. 같은팀의2025제지RL논문/2026안전영역논문은 별개의연구|

2025AISP 사이트는 전문을1주일 게시한다고 안내하지만 현재 페이지에는 논문 PDF 링크가 없다. 공개 GitHub 저장소 `bpai2025/bpai2025.github.io`의 모든34개commit에서 `index.html` PDF링크를 역추적했다. 실제 Camera-ready 링크를 발견하지 못했고 arXiv/supplementary 링크는 사이트템플릿 placeholder였다. 현재HEAD의 `static/pdfs/sample.pdf`는 수상논문으로 취급하지 않는다. 2026repo는 공식 양식DOCX/HWP와sample만있다. 공식학회 논문집 다운로드는 이메일인증으로, 경진대회 제출은 로그인으로 이동한다. 해당 인증/유료접근을 우회하지 않았다.

### 공개 전문 분석: Conditional Flow Policy와 연결된 별도 로봇 논문

[Flow-Guided Policies: Overcoming Diffusion Limitations for Robust Robot Imitation Learning](https://openaccess.thecvf.com/content/ICCV2025W/ACVR/papers/Jung_Flow-Guided_Policies_Overcoming_Diffusion_Limitations_for_Robust_Robot_Imitation_Learning_ICCVW_2025_paper.pdf), ICCVW2025 공개6p를 직접 읽었다. 이는 경진대회의4p 원고가 아니며, 그 논문을 reference[9]로 인용하는 후속/관련논문이다. 수상작 본문을 확보한 것으로 세지 않는다. PDF 상대페이지와 인쇄페이지를 함께 표기한다.

- p1(2507): 문제·DP의 제약·제안policy·주장을 Abstract/Introduction에 배치. p2(2508): 관련연구를 BC/생성정책/diffusion/flow로 분류. p3(2509): 관측history→action chunk와 ODE/flow 수식, Fig1 구조도. p4(2510): 학습손실과 동일네트워크/encoder/optimizer 설정, Table1 task별성공률, Fig2 정성비교. p5(2511): 평가설정·실물예시·Table2 solver비용/성공, samplingstep그래프. p6(2512): sampling효율해석·결론·사사·참고문헌.
- 비교근거: p5 §4.1–4.2가 동일expert demonstration과 표준화 평가, **task당50rollout의 binary success 평균** 및 BC-RNN/DP를 명시한다. p4Table1은 기본/contact-rich/long-horizon을 나눠13task를 나란히 놓는다. 특정실패영상만 제시하지 않고 task표로 적용범위를 드러낸다는 점을 참고할 수 있다.
- 보조평가: p5Table2는 Euler/Midpoint/Dopri5/AdaptiveHeun의 성공score와 latency(ms)를 함께 제시한다. p5Fig4~p6§4.4는 samplingstep을 바꿔 정확도·계산비용 tradeoff를 본다. 이것은 component-removal ablation과 달라 solver/step 민감도라고 부른다.
- 논문 자체의 한계도 있다. p5§4.2는 모든task에서 최고라고 서술하지만 p4Table1은 StackThree에서 BC-RNN0.86>CoF0.80, NutAssembly DP0.64>CoF0.60이다. 따라서 서술을 따라 과장하지 않고 표를 확인해야 한다. 학습seed수·CI·latency 측정장비는 읽은본문에서 명확하지 않아 재현성이 완전하다고 평가하지 않는다.
- 우리 원고에 적용할 편집 판단(수상기준이 아닌 판단): 고장이라는 문제→수식화→구조도→같은조건baseline표→관절/열화별실패범위→연산/학습세부·제약으로 연결한다. safety와mobility를 분리하고 worst-condition까지 보이는 표가 핵심이다. 50rollout이라는 수를 그대로베끼지 않고 실제평가표본수·seed를 명시한다.

### PIM 수상작과 연결된 후속 논문의 공개 전문

[JSTS Hardware-software Co-design for Vector Similarity Search on HBM-PIM](https://www.jsts.org/jsts/XmlViewer/f449809), DOI10.5573/JSTS.2025.25.6.662는 공개HTML 전문을 확인했다. 2024수상5p 원고와 동일문서가 아니며 저자구성/연도/확장내용이 다르다. PDF링크의 AURIC 경로는 'not authorized user' 응답이라 PDF페이지단위 분석은 하지 않았다. 이 후속논문 내용을2024수상본문 실험으로 소급하지 않는다.

전문을 확보하지 못한 대상의 baseline·metric·ablation·reproducibility를 꾸며 작성하지 않는다. 현재 수상원문의 실제읽기 기반 종합분석은 **미완료**이며, 정확한논문DB기록과 접근경로를 제공해 사용자 학교기관구독/저자공개본이 확보되면 이어갈 수 있도록 남겼다.
