# 논문용 그림 목록

현재 원고는 [생존율+네 RMSE 합본 PDF](paper_performance_compact.pdf)와 Student 절대 낙상률 히트맵을 사용한다. 아래 개별 곡선도 독립 파일로 제공한다.

| 그림 | PDF | PNG | 실제 크기 |
|---|---|---|---|
| 생존율 | [PDF](paper_survival_by_degradation.pdf) | [PNG](paper_survival_by_degradation.png) | 80 × 50 mm |
| vx RMSE | [PDF](paper_vx_rmse_by_degradation.pdf) | [PNG](paper_vx_rmse_by_degradation.png) | 80 × 50 mm |
| vy RMSE | [PDF](paper_vy_rmse_by_degradation.pdf) | [PNG](paper_vy_rmse_by_degradation.png) | 80 × 50 mm |
| yaw-rate RMSE | [PDF](paper_yaw_rate_rmse_by_degradation.pdf) | [PNG](paper_yaw_rate_rmse_by_degradation.png) | 80 × 50 mm |
| 경로 횡오차 RMSE | [PDF](paper_cross_track_rmse_by_degradation.pdf) | [PNG](paper_cross_track_rmse_by_degradation.png) | 80 × 50 mm |
| Student 관절별 낙상률 | [PDF](paper_student_fall_rate_heatmap.pdf) | [PNG](paper_student_fall_rate_heatmap.png) | 80 × 85 mm |

모든 현재 그림 파일에는 제목을 넣지 않는다. 그림 번호·제목·캡션은 원고에서 지정한다. 다섯 곡선은 각각 축·단위·범례를 포함하며 Times New Roman 9 pt다. 히트맵은 단일 청색·얇은 격자의 한 단 세로형이다. 음영은 평가 시드 3개의 평균값 최소–최대다.

히트맵은 세로 12관절, 가로 20–100% 토크 저하율이다. 셀당 Student 216시행의 낙상 비율을 0–30% 고정 색 범위와 소수점 한 자리로 표시한다. [정확한 셀별 낙상 수와 비율](paper_student_fall_rate_heatmap.csv)도 제공한다. 생존은 완주 또는 비낙상 상한 종료이며, 정체한 비낙상도 포함한다.

이 그림이 보여줄 것은 “어느 고장 관절과 저하율에서 Student의 낙상률이 얼마인가?”다. Normal과의 차이는 앞의 곡선에서 비교하고, 추종·정체는 RMSE와 사례를 함께 해석한다. 지형·과제 비율에 따른 효과를 이 그림으로 주장하지 않는다.

`paper_joint_vxy_gain.*`와 `paper_metrics_by_degradation.*`는 이전 차이 히트맵·합본 시안으로 보존한 파일이다. 현재 논문에 넣을 그림은 위 목록이다. `terrain_heightmap_s1101.*`는 데이터의 지형 설명용 보조 자료다.

2026-10-17: 현재 히트맵은 낙상률(100−생존율), 낮은 값은 밝고 높은 값은 진하다. `paper_student_survival_heatmap.*`은 이전 생존율 시안으로 보존하며 현재 원고에서는 사용하지 않는다. 데이터·생존율 정의는 바꾸지 않았다.

최신 원고 편집: 생존율 위 한 패널과 네 RMSE의 2×2 배치인 `paper_performance_compact.pdf`(80×107 mm)를 사용한다. 독립 곡선 파일은 유지한다. 낙상률 색 범위는 0–30%이며 값은 변경하지 않았다. 현재 원고는 프로시딩6쪽·익명6쪽으로 목표5/4쪽 미충족이다.
