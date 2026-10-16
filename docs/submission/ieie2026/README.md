# 2026-10-17 최신 원고

[용어, 문단, 그림 설명 정리](TERMINOLOGY_20261017.md): 최신 용어와 표 6의 목적, 캡션 정리 및 검증 결과.

[편집기 PDF 참조·인용 오류 수정 및 전체 검사](REFERENCE_FIX_20261017.md): 루트 설정의 XeLaTeX 충돌을 수정하고 canonical build와 preview를 동기화했다.

최신 편집본은 [프로시딩 6쪽](private/latex/preview/proceedings.pdf)과 [익명 심사용 6쪽](private/latex/preview/main.pdf)이다. [서론·그림·결과 편집 기록](EDITORIAL_20261017.md)을 확인한다. 목표 5쪽·4쪽은 아직 미충족이다. [이전 복구·리뷰](RESTORATION_20261017.md)는 직전 7쪽·6쪽 원고에 대한 기록이다.

# 대한전자공학회 2026 추계 학부생 논문경진대회 준비

최종 정책은 JT77500 Student-only이며, 연구 범위는 Isaac Gym에서 단일 관절의 PD 토크 계수 저하에 대응하는 4족보행이다. 기존 발표의 학습 이력 비교를 논문의 주요 경쟁 방법 비교로 사용하지 않는다.

## 현재 논문과 노트북 편집

프로시딩 5쪽을 기본 원고로 작성하고, 핵심 내용으로 줄인 익명 심사용 4쪽을 별도 관리한다.

- [프로시딩 본문 body.tex](private/latex/body.tex)
- [심사용 본문 body_review.tex](private/latex/body_review.tex)
- [VS Code 작업 영역](private/latex/ieie-paper.code-workspace)
- [설정·빌드 방법](private/latex/README.md)
- [최신 프로시딩 PDF](private/latex/preview/proceedings.pdf), [최신 심사용 PDF](private/latex/preview/main.pdf)
- [최종 발표 자료](../../present_first/README.md)

사용자 요청으로 현재 TeX·필수 이미지·최종 미리보기를 Git에 포함한다. 이전 초안과 중간 출력은 제거했다. 고정 비율 2:1:1:2 평가의 원자료와 상세 조건은 `datasets/ieie2026_fixed2112/`에 별도 제공하며 대용량 원자료는 Git LFS로 관리한다. 체크포인트, 다운로드한 제3자 논문, 폰트 파일과 나머지 private 연구 자료는 제외한다.

## 연구 근거

`rules_and_templates.md`는 공식 조건과 양식, `related_work_metrics.md`는 선행 연구와 지표 비교를 기록한다. 수치의 원본 평가 자료는 로컬 `results/`에 보존한다. 기존 Word 제작 스크립트는 현재 TeX를 편집하거나 변환하는 경로가 아니다.

## 연구의 중심 질문

> 단일 관절의 토크가 저하된 뒤에도, 고장대응 정책은 정상학습 정책보다 다양한 이동 명령과 목표 경로를 더 오래, 더 정확하게 수행할 수 있는가?

생존율, 이동 명령 추종 오차, 경로 횡오차를 함께 평가한다. 생존율은 평가 종료까지 낙상하지 않은 시행의 비율이며, 추종 오차는 고장 발생 후 종료 전 유효 구간의 시행별 RMSE 평균이다. 완주율과 관찰된 정체 사례를 통해 실제 진행 여부도 살펴본다. 고정된 두 정책 구성의 비교이며, 평가 시드 세 개를 독립 학습 반복으로 해석하지 않는다.
