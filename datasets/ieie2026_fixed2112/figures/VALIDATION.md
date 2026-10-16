# 개별 그림과 Student 낙상률 히트맵 검증

- 현재 원고는 `paper_performance_compact.pdf`와 Student 절대 낙상률 히트맵을 사용한다. 독립 곡선도 제공한다.
- 개별 곡선은 각각 80 × 50 mm, 히트맵은 80 × 85 mm다. Times New Roman을 사용하며 축·눈금·범례·셀 숫자는 실제 삽입 크기에서 9 pt다.
- 다섯 곡선의 표시 수치를 원래 집계 CSV와 대조했다. 최대 차이는 7.22e-16으로 부동소수점 반올림 범위다.
- Student 히트맵 60셀의 낙상 수와 분모 216이 기존 관절별 요약과 일치한다. 색 범위는 0.4–30% 로그 스케일다.
- 셀 숫자와 배경의 최소 계산 대비는 14.01:1이며, 60셀 모두 숫자가 경계 안에 들어간다. 모든 현재 그림에서 글자가 바깥으로 넘치지 않는다.
- 일곱 현재 PDF의 Times New Roman 일반체·수식 이탤릭체 임베딩을 확인했다. 새 히트맵과 곡선 PNG를 직접 열어 확인했다.
- 모든 현재 그림은 내부 제목이 없다. 번호·제목·캡션은 원고에서 별도로 지정한다.
- 곡선 음영은 평가 시드 3개 평균의 최소–최대이며 학습 시드 신뢰구간이 아니다.
- 근거는 `style_audit.json`, `paper_student_fall_rate_heatmap.csv`, `../reproducibility/individual_figures_validation.json`이다.
- 기존 `paper_joint_vxy_gain.*`·`paper_metrics_by_degradation.*`는 이전 시안으로 보존한다. 현재 본문 추천이 아니다.
- 실제 원고에서 그림·캡션·쪽수를 함께 검수했으며 결과는 개정 기록에 있다.

- 이 그림들을 복구한 6쪽 프로시딩·6쪽 익명 원고에 삽입해 검수했다. 목표 5쪽·4쪽은 미충족이다. 원고와 전체 검증은 `../../../docs/submission/ieie2026/EDITORIAL_20261017.md`를 참조한다.

2026-10-17: 현재 히트맵은 낙상률(100−생존율), 낮은 값은 밝고 높은 값은 진하다. `paper_student_survival_heatmap.*`은 이전 생존율 시안으로 보존하며 현재 원고에서는 사용하지 않는다. 데이터·생존율 정의는 바꾸지 않았다.

최신 원고 편집: 생존율 위 한 패널과 네 RMSE의 2×2 배치인 `paper_performance_compact.pdf`(80×107 mm)를 사용한다. 독립 곡선 파일은 유지한다. 낙상률 색 범위는 0.4–30% 로그 스케일이며 값은 변경하지 않았다. 현재 원고는 프로시딩6쪽·익명6쪽으로 목표5/4쪽 미충족이다.

현재 TeX의 그림4는 `paper_survival_compact.pdf`(80×38 mm), 그림5는 `paper_tracking_rmse_2x2.pdf`(80×74 mm), 그림6은 낙상률 히트맵이다. 히트맵 숫자는 그대로 두고 색에만 LogNorm(0.4,30)을 적용했다. `paper_performance_compact.*`는 직전 합본으로 보존하며 현재 원고는 사용하지 않는다.
