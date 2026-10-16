# 개별 그림과 Student 생존율 히트맵 검증

- 현재 본문 후보는 `README.md`의 다섯 개별 곡선과 Student 절대 생존율 히트맵이다.
- 개별 곡선은 각각 80 × 65 mm, 히트맵은 80 × 100 mm다. Times New Roman을 사용하며 축·눈금·범례·셀 숫자는 실제 삽입 크기에서 9 pt다.
- 다섯 곡선의 표시 수치를 원래 집계 CSV와 대조했다. 최대 차이는 7.22e-16으로 부동소수점 반올림 범위다.
- Student 히트맵 60셀의 생존 수와 분모 216이 기존 관절별 요약과 일치한다. 색 범위는 0–100%다.
- 셀 숫자와 배경의 최소 계산 대비는 4.60:1이며, 60셀 모두 숫자가 경계 안에 들어간다. 모든 현재 그림에서 글자가 바깥으로 넘치지 않는다.
- 일곱 현재 PDF의 Times New Roman 일반체·수식 이탤릭체 임베딩을 확인했다. 새 히트맵과 곡선 PNG를 직접 열어 확인했다.
- 모든 현재 그림은 내부 제목이 없다. 번호·제목·캡션은 원고에서 별도로 지정한다.
- 곡선 음영은 평가 시드 3개 평균의 최소–최대이며 학습 시드 신뢰구간이 아니다.
- 근거는 `style_audit.json`, `paper_student_survival_heatmap.csv`, `../reproducibility/individual_figures_validation.json`이다.
- 기존 `paper_joint_vxy_gain.*`·`paper_metrics_by_degradation.*`는 이전 시안으로 보존한다. 현재 본문 추천이 아니다.
- 전체 원고에 삽입한 뒤의 쪽수·캡션 간격·조판 검수는 별도 작업이다.
