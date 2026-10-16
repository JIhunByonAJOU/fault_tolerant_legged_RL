# 2026-10-16 최신 원고

새 혼합 과제 평가를 반영한 [프로시딩 5쪽](private/latex/preview/proceedings.pdf)과 [익명 심사용 4쪽](private/latex/preview/main.pdf)을 제공한다. [개정·검증 기록](REVISION_20261016.md), [AI 독립 리뷰 및 수정 대응표](reviewer/revision_20261016/RESPONSE.md)를 먼저 확인한다. 아래 내용 중 이전 직진 평가·Q 관련 기록은 과거 작업이다.

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

사용자 요청으로 현재 TeX·필수 이미지·최종 미리보기를 Git에 포함한다. 이전 초안과 중간 출력은 제거했다. 원본 평가 로그·체크포인트·다운로드한 제3자 논문·폰트 파일과 나머지 private 연구 자료는 계속 제외한다.

## 연구 근거

`rules_and_templates.md`는 공식 조건과 양식, `related_work_metrics.md`는 선행 연구와 지표 비교를 기록한다. 수치의 원본 평가 자료는 로컬 `results/`에 보존한다. 기존 Word 제작 스크립트는 현재 TeX를 편집하거나 변환하는 경로가 아니다.

## 연구의 중심 질문

> 주행 중 한 관절의 토크 계수가 낮아졌을 때, 고장 정보를 직접 제공받지 않는 관측 이력 기반 정책은 정상 정책보다 얼마나 오래 생존하고, 얼마나 전진하며, 목표 속도를 얼마나 정확하게 유지하는가?

안전성만 높은 정책을 좋은 보행 정책으로 단정하지 않는다. 생존율과 alive-time, 목표 대비 전진량, 정지/후진 시간, 연속 속도/yaw 오차, 엄격 추종 충족 시간을 함께 해석한다. 기록되지 않은 지표를 기존 평균값으로 복원할 수 있다고 가정하지 않는다.
