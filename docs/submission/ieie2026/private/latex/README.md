# 논문 LaTeX 편집과 최종 개정본

2026-10-17에 `0d34639`의 서론·시스템·방법과 사진·표 1–4를 복구하고 최신 26,352시행 평가를 연결했다. **현재 프로시딩 7쪽, 익명 심사용 6쪽으로 목표 5쪽·4쪽은 미충족**이다. 두 PDF는 실제 LuaLaTeX 빌드를 마쳤다. [복구·검증 기록](../../RESTORATION_20261017.md)을 기준으로 사용한다.

| 용도 | 편집 파일 | 진입 파일 | 최신 PDF |
|---|---|---|---|
| 프로시딩 | [body.tex](body.tex) | [proceedings.tex](proceedings.tex) | [preview/proceedings.pdf](preview/proceedings.pdf) |
| 익명 심사용 | [body_review.tex](body_review.tex) | [main.tex](main.tex) | [preview/main.pdf](preview/main.pdf) |

두 원고 모두 로봇·환경 사진, 시스템·관측·인코더·보상표, 이전 구조도와 학습 곡선을 포함한다. 프로시딩은 다섯 개별 성능 곡선과 Student 절대 낙상률 히트맵을 사용한다. 심사용은 세 축 명령 오차를 표로 제시한다. 익명본에서는 저자·소속·이메일과 저자를 식별하는 GitHub 링크를 제외했다.

## 내용과 근거

- 생존율은 낙상 여부의 비율이다. 정규화 생존 시간 또는 완주율이 아니다.
- 명령·경로 RMSE는 고장 후 종료 전 유효 표본에서 시행별 계산 후 평균한다. 낙상 뒤를 0으로 채우지 않는다.
- 정체는 결과 해석의 짧은 보조 사례이고 결론의 중심 주장이 아니다. 완주 종료 비율은 기존 기록을 재집계해 보조 확인으로 추가했다.
- [원자료·분석](../../../../../datasets/ieie2026_fixed2112/README.md), [수치·설정·학습 계보·조건별 표](../../paper_assets/MANUSCRIPT_EVIDENCE.md), [검증 JSON](../../paper_assets/manuscript_validation.json)을 함께 제공한다.
- [리뷰·수정 대응표](../../reviewer/restoration_20261017/RESPONSE.md)는 대화 맥락 없는 교수·로보틱스·박사생 역할 AI의 첫 심사와 수정 후 재심을 보존한다. 실제 외부 전문가 승인으로 표시하지 않는다.

## Ubuntu와 Windows 빌드

저장소 루트의 `ieie-paper.code-workspace`를 열고 `dev` 브랜치 최신 파일을 받는다. Git LFS 자료까지 받으려면 원자료 README의 명령을 따른다. 그림만 포함해 원고를 빌드하는 데 원시 trace 전체나 Isaac Gym은 필요 없다.

```sh
cd docs/submission/ieie2026/private/latex
latexmk -lualatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build proceedings.tex
latexmk -lualatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build main.tex
```

Ubuntu 패키지가 필요하면 `latexmk`, `texlive-luatex`, `texlive-lang-korean`, `texlive-latex-extra`, `texlive-fonts-recommended`, `fonts-noto-cjk`를 설치한다. Windows에서는 TeX Live 또는 MiKTeX와 LuaLaTeX, latexmk를 PATH에서 실행할 수 있어야 한다. 빌드 결과는 Git 제외 경로 `build/`에 생성된다. 편집 후 PDF 미리보기는 직접 빌드한 파일을 사용한다.

## 글꼴과 그림

- `fonts-proceedings.tex`는 설치된 HY신명조를 우선하며, 제목 바탕·저자 한컴바탕·수식 Cambria Math 설정을 사용한다. 시작 시 이미 있던 OS 공통 글꼴 선택 수정을 보존해 현재 빌드 설정으로 포함했다. 글꼴 바이너리는 배포하지 않는다.
- 동일 글꼴이 없는 PC는 편집용 대체 글꼴을 사용하므로 쪽수·배치가 달라질 수 있다. 현재 PDF는 본문 글자·여백을 줄여 한도에 맞춘 결과가 아니다.
- 그래프는 Times New Roman 9 pt이며 PDF 안에 글꼴을 포함한다. 성능 곡선 80×50 mm, 히트맵 80×85 mm, 학습 곡선은 80×40 mm다. 복구한 이전 구조도 `paper_workflow_v8.pdf`는 원래 서체·배치를 보존했다. 내부 제목 없이 원고 캡션으로 구별한다.
- `preamble.tex`의 `graphicspath`는 저장소의 `datasets/ieie2026_fixed2112/figures/`와 `docs/submission/ieie2026/paper_assets/`를 직접 참조한다. 폴더 구조를 그대로 유지한다.
- [그림 생성기](../../make_manuscript_figures.py)는 보존한 학습 CSV로 학습 곡선과 이전 단순화 구조도 시안을 생성한다. 현재 원고의 복구 구조도는 `private/assets/paper_workflow_v8.pdf`이며 생성기가 덮어쓰지 않는다. 성능 곡선은 데이터 폴더의 `paper_figures.py`를 사용한다.

공식 양식은 DOCX/HWP이며 이 TeX는 공식 LaTeX 양식이 아니다. 기존에 측정한 공식 양식의 글꼴·여백·줄 간격을 보존했다. 같은 글꼴 환경에서 최종 PDF를 확인한다. 사사 문구·과제번호는 제공되지 않아 임의로 만들지 않았으며 저자 자격과 함께 제출 전에 확인해야 한다.
