# 논문 그림 제작 기준과 재현 방법

[조교 피드백](TA_GUIDANCE.md)의 작은 글씨·어두운 셀 숫자·표기 불일치를 해결하는 제작 기준이다. 글꼴 이름과 9 pt는 프로젝트가 정한 값이며 조교가 명시한 숫자가 아니다.

## IEEE 공식 안내에서 확인한 기준

[IEEE Author Center의 공식 그림 안내](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/improve-your-graphics/)를 2026-10-16에 확인했다. 안내는 Times New Roman 등의 글꼴, 실제 크기에서 약 9–10 pt, 일관된 서체·크기, PDF 폰트 임베딩, 색 이외의 선·표식 사용을 권고한다. 이 안내에서 특정 팔레트나 셀 모양을 단일 ‘IEEE 히트맵 스타일’로 지정하지는 않는다.

이번 제작은 해당 가독성 원칙을 참고한다. Times New Roman 9 pt로 영문 그래프를 통일하고, 실제 원고 한 단 폭인 80 mm를 기준으로 제작한다. 열 폭이 다른 IEEE 원고의 수치를 대한전자공학회 원고에 그대로 적용하지 않는다.

## 공통 스타일

| 요소 | 기준 | 확인 방법 |
|---|---|---|
| 글꼴 | Times New Roman, 축·눈금·범례·셀 숫자 모두 동일 | 생성한 `style_audit.json`과 PDF의 실제 임베딩 폰트 확인 |
| 글자 크기 | 최종 삽입 크기에서 9 pt | 개별 곡선은 처음부터 80 mm 폭·9 pt로 만든다. 곡선·히트맵 모두 처음부터 80 mm 폭에서 9 pt로 제작한다 |
| 그림 폭 | 개별 곡선은 80 × 65 mm, 히트맵은 80 × 100 mm | 현재 프로시딩 양식 분석의 본문 폭 약 170 mm와 맞춘 제작값. 최종 TeX의 `textwidth`를 다시 확인 |
| 곡선 | Normal은 주황 점선·원 표식, Student는 파랑 실선·사각 표식 | 흑백에서도 선·표식으로 구별 |
| 축 | 모든 저하율 곡선에 0, 20, 40, 60, 80, 100% 표시 | 정상 조건을 특정 관절에 12번 복제하지 않음 |
| 단위 | vx/vy: m/s, yaw-rate: rad/s, 횡오차: m, 생존율: % | 서로 다른 물리 단위를 한 점수로 합치지 않음 |
| 파일 | 벡터 PDF와 300 dpi PNG | 논문에는 PDF, 빠른 열람에는 PNG 사용 |
| 제목·캡션 | 그림 파일 내부에는 없음 | 번호·제목·캡션은 원고에서 별도 지정. 축·단위·범례는 유지 |

## 본문 히트맵 한 장

선택한 지표는 **Student의 절대 생존율(%)**이다. 이 그림의 질문은 “제안 정책이 어느 관절의 어느 토크 저하에서 얼마나 낙상을 피하는가?”다. Normal 대비 비교는 다섯 개별 곡선에서 하고, 히트맵은 전체 평균에 가려진 Student의 관절별 취약성을 보여준다.

- 세로 12관절 × 가로 5저하율. 각 셀은 Student 216시행의 생존 비율이다.
- 값은 `100 * survived / episodes`. 색은 0–100% 고정 범위의 단일 청색(Blues)이며 차이·증가량이 아니다. 80 × 100 mm의 세로형에 얇은 경계선과 좁은 색 막대를 쓴다.
- 셀 숫자는 소수점 한 자리, 색 막대 단위는 %. 정확한 생존 수·분모는 `paper_student_survival_heatmap.csv`에 저장한다.
- 배경색의 실제 밝기에 따라 흰색/검은색 숫자를 선택하고 60셀의 텍스트 경계를 검사한다.
- 정상 조건은 특정 고장 관절이 없으므로 히트맵에 12번 복제하지 않는다.
- 생존율은 낙상 여부이며 정체한 비낙상도 포함한다. 명령·경로 수행은 네 RMSE 곡선과 정체 사례로 함께 해석한다.

시안 캡션:

> Student survival rate by degraded joint and torque degradation. Each cell reports the percentage of 216 trials that terminated at the goal or administrative cap without a fall. Motion-tracking errors and stagnation are evaluated separately.

100% 저하에서 FR thigh 71.76%, FL thigh 78.70%, FR hip와 RR thigh 각각 97.22%다. 이 수치 차이를 근거로 같은 토크 저하에서도 제안 정책의 생존 성능이 관절에 따라 달라진다고 설명한다. 지형이나 과제 비율의 효과를 이 그림으로 주장하지 않는다.

이전 평면속도 RMSE 차이 히트맵과 5패널 진단 히트맵은 과거 시안·보조 자료로 보존한다. 현재 본문 추천은 Student 생존율 히트맵 한 장이다.

## 저하율별 다섯 곡선

- 생존율, vx RMSE, vy RMSE, yaw-rate RMSE, 경로 횡오차 RMSE를 **각각 별도 PDF·PNG**로 저장한다. 각 파일에 축·단위·Normal/Student 범례를 넣는다. 합본 패널을 잘라 쓰지 않는다.
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
py paper_figures.py --font "C:\Windows\Fonts\times.ttf"
```

다른 글꼴로 변경하기로 정했다면 `--font-family` 또는 `--font`를 **전체 그림에 한 번에** 적용한다. 현재 시안은 `figures/`에, 재생성 결과는 기본적으로 Git 제외 경로 `generated/paper_figures/`에 저장한다.

`style_audit.json`에는 폰트 경로·SHA-256·그림 크기·실제 글자 크기·셀 대비·셀 경계·내부 제목 없음 검사 결과를 저장한다. 생성 시안의 가독성 확인은 전체 원고의 캡션 간격·익명성·5/4쪽 제한 검수를 대체하지 않는다.
