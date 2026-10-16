# 새 그림 시안 검증

- 공통 글꼴: HYSinMyeongJo-Medium. 실제 PDF 내부 이름은 H2mjsM이며 `pdffonts`에서 세 PDF 모두 CID TrueType, embedded=yes, subset=yes로 확인했다.
- 곡선: 170 × 112 mm, 히트맵: 170 × 95 mm, 지형 설명 그림: 170 × 125 mm. 논문에 넣을 때 이 실제 폭을 유지한다.
- 축·눈금·범례·셀 숫자의 생성 크기는 모두 9 pt다.
- 히트맵 60셀의 숫자 모두 각 셀 안에 들어온다. 그림 밖으로 벗어나는 텍스트는 없다.
- 셀 숫자와 배경 사이의 최소 계산 대비는 약 5.13:1이다. PNG를 실제로 열어 숫자·축·범례·부호를 확인했다.
- 저하율별 곡선의 원자료 재집계와 기존 CSV 차이는 최대 약 7.22e-16으로 부동소수점 반올림 범위다. 근거는 `../reproducibility/documentation_validation.json`에 있다.
- 음영은 3개 평가 시드 평균의 최소–최대이며 신뢰구간이 아니다.
- 조교 메시지에는 특정 폰트·pt 값이 없었다. HY신명조·9 pt는 현재 원고에 맞춘 제작 기준이다.
- `paper_joint_vxy_gain.pdf`와 `paper_metrics_by_degradation.pdf`는 본문 삽입 후보, `terrain_heightmap_s1101.pdf`는 데이터의 지형 설명용이다. 후자는 x/y 축 배율이 다름을 제목에 표시한다.
- 전체 TeX에 삽입한 뒤의 쪽수·캡션 간격·조판 검수는 아직 별도 작업이다.
