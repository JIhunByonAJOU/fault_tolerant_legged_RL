# 논문 그림 제작 기준과 재현 방법

[조교 피드백](TA_GUIDANCE.md)의 작은 글씨·어두운 셀 숫자·표기 불일치를 해결하는 제작 기준이다. 글꼴 이름과 9 pt는 프로젝트가 정한 값이며 조교가 명시한 숫자가 아니다.

## 공통 스타일

| 요소 | 기준 | 확인 방법 |
|---|---|---|
| 글꼴 | HY신명조, 축·눈금·범례·셀 숫자 모두 동일 | 생성한 `style_audit.json`과 PDF의 실제 임베딩 폰트 확인 |
| 글자 크기 | 최종 삽입 크기에서 9 pt | 시안의 실제 폭 170 mm를 유지한다. 80 mm 한 단에 그대로 축소하면 약 4.2 pt가 되므로 금지 |
| 그림 폭 | 주결과 곡선과 히트맵은 두 단 폭 170 mm | 현재 프로시딩 양식 분석의 본문 폭 약 170 mm와 맞춘 제작값. 최종 TeX의 `textwidth`를 다시 확인 |
| 곡선 | Normal은 주황 점선·원 표식, Student는 파랑 실선·사각 표식 | 흑백에서도 선·표식으로 구별 |
| 축 | 모든 저하율 곡선에 0, 20, 40, 60, 80, 100% 표시 | 정상 조건을 특정 관절에 12번 복제하지 않음 |
| 단위 | vx/vy: m/s, yaw-rate: rad/s, 횡오차: m, 생존율: % | 서로 다른 물리 단위를 한 점수로 합치지 않음 |
| 파일 | 벡터 PDF와 300 dpi PNG | 논문에는 PDF, 빠른 열람에는 PNG 사용 |
| 캡션 | 그림 밖의 TeX 캡션 | 데이터·부호·분모·오차 구간은 캡션 또는 본문에 설명 |

## 본문 히트맵 한 장

선택한 지표는 **평면 이동 명령 RMSE의 개선량**이다. 각 시행에서

`E_vxy = sqrt((SSE_vx + SSE_vy) / valid_post_samples)`

를 구한 뒤 각 관절·저하율 셀에서 정책별 평균을 내고 `Normal - Student`를 표시한다. 이는 `sqrt(RMSE_vx² + RMSE_vy²)`와 같다. 평균 RMSE 두 개를 나중에 합치지 않는다. yaw-rate는 포함하지 않는다.

- 12관절 × 5저하율. 셀당 각 정책 216시행.
- 0 중심의 대칭 색 범위. 양수 파랑은 Student의 오차가 작다는 의미, 음수 빨강은 반대다.
- 숫자는 부호 포함 소수점 세 자리. CSV는 원래 정밀도를 보존한다.
- 배경색을 sRGB에서 선형 밝기로 변환한 뒤, 검은색·흰색 중 대비가 큰 색을 선택한다. 절대값 임계치만으로 글자색을 정하지 않는다.
- 생성기에서 숫자의 렌더링 영역이 셀 경계 안에 들어오는지 확인한다. 60셀 모두 검증한다.
- 최종 실제 크기로 인쇄하거나 PDF 100% 배율로 읽어 본다. 색만으로 설명하지 않고 숫자·색 막대·부호를 함께 쓴다.

시안 캡션:

> Reduction in planar velocity-command RMSE (Normal − Student) by degraded joint and torque degradation. Positive values indicate lower error for Student. Each cell averages 216 trials per policy over the fixed 2:1:1:2 task mixture, 12 terrain columns, and three evaluation seeds. Trial errors use valid post-fault samples before termination.

생존율·vx·vy·yaw-rate·횡오차의 기존 5패널 히트맵은 보조 진단 자료다. 본문에 이들을 모두 삽입하는 계획이 아니다.

## 저하율별 다섯 곡선

- 같은 행·열의 5패널에서 생존율과 네 오차를 각각 표시한다. 서로 다른 지표의 y축을 공유하지 않는다.
- 각 점은 시행별 지표의 평균이며, 임의로 곡선을 매끈하게 보간하지 않는다.
- 음영은 3개 **평가 시드 평균의 최소–최대**다. 학습 반복의 신뢰구간 또는 95% 신뢰구간이라고 쓰지 않는다.
- 100%뿐 아니라 정상·낮은 저하 조건도 포함한다. 모든 지표가 단조롭게 나빠지거나 Student가 전 조건에서 우수하다는 인상을 만들지 않는다.

## 정체와 대표 시계열

본문에는 정체 히트맵을 추가하지 않고 RL calf 사례를 짧은 표와 문장으로 설명한다. 필요하면 대응 시행 시계열을 한 그림으로 넣는다. 시계열은 실제값과 각 정책에 실제로 전달된 명령을 구분하고, 낙상·완주 후 선을 0으로 연장하지 않는다. 탑다운은 독립적인 기준 경로와 실제 궤적을 표시한다. 본문 그림 수가 부족하면 시계열은 보조 자료로 둔다.

## 실행

Windows에서 이 폴더로 이동한 뒤:

```powershell
py -m pip install -r requirements.txt
py paper_figures.py
```

기본 글꼴을 찾지 못하면 자동으로 다른 글꼴로 바꾸지 않고 중단한다. 설치된 파일을 직접 지정할 수 있다.

```powershell
py paper_figures.py --font "C:\Windows\Fonts\H2MJSM.TTF"
```

다른 글꼴로 변경하기로 정했다면 `--font-family` 또는 `--font`를 **전체 그림에 한 번에** 적용한다. 현재 시안은 `figures/`에, 재생성 결과는 기본적으로 Git 제외 경로 `generated/paper_figures/`에 저장한다.

`style_audit.json`에는 폰트 경로·SHA-256·그림 크기·실제 글자 크기·셀 대비·셀 경계 검사 결과를 저장한다. 생성 시안의 가독성 확인은 전체 원고의 캡션 간격·익명성·5/4쪽 제한 검수를 대체하지 않는다.
