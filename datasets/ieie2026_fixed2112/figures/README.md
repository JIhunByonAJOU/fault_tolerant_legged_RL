# 논문용 그림 목록

현재 추천은 **다섯 개의 개별 비교 곡선 + Student 절대 생존율 히트맵 한 장**이다.

| 그림 | PDF | PNG | 실제 크기 |
|---|---|---|---|
| 생존율 | [PDF](paper_survival_by_degradation.pdf) | [PNG](paper_survival_by_degradation.png) | 80 × 65 mm |
| vx RMSE | [PDF](paper_vx_rmse_by_degradation.pdf) | [PNG](paper_vx_rmse_by_degradation.png) | 80 × 65 mm |
| vy RMSE | [PDF](paper_vy_rmse_by_degradation.pdf) | [PNG](paper_vy_rmse_by_degradation.png) | 80 × 65 mm |
| yaw-rate RMSE | [PDF](paper_yaw_rate_rmse_by_degradation.pdf) | [PNG](paper_yaw_rate_rmse_by_degradation.png) | 80 × 65 mm |
| 경로 횡오차 RMSE | [PDF](paper_cross_track_rmse_by_degradation.pdf) | [PNG](paper_cross_track_rmse_by_degradation.png) | 80 × 65 mm |
| Student 관절별 생존율 | [PDF](paper_student_survival_heatmap.pdf) | [PNG](paper_student_survival_heatmap.png) | 80 × 100 mm |

모든 현재 그림 파일에는 제목을 넣지 않는다. 그림 번호·제목·캡션은 원고에서 지정한다. 다섯 곡선은 각각 축·단위·범례를 포함하며 Times New Roman 9 pt다. 히트맵은 단일 청색·얇은 격자의 한 단 세로형이다. 음영은 평가 시드 3개의 평균값 최소–최대다.

히트맵은 세로 12관절, 가로 20–100% 토크 저하율이다. 셀당 Student 216시행의 생존 비율을 0–100% 고정 색 범위와 소수점 한 자리로 표시한다. [정확한 셀별 생존 수와 비율](paper_student_survival_heatmap.csv)도 제공한다. 생존은 완주 또는 비낙상 상한 종료이며, 정체한 비낙상도 포함한다.

이 그림이 보여줄 것은 “어느 고장 관절과 저하율에서 Student의 생존율이 얼마인가?”다. Normal과의 차이는 앞의 곡선에서 비교하고, 추종·정체는 RMSE와 사례를 함께 해석한다. 지형·과제 비율에 따른 효과를 이 그림으로 주장하지 않는다.

`paper_joint_vxy_gain.*`와 `paper_metrics_by_degradation.*`는 이전 차이 히트맵·합본 시안으로 보존한 파일이다. 현재 논문에 넣을 그림은 위 목록이다. `terrain_heightmap_s1101.*`는 데이터의 지형 설명용 보조 자료다.
