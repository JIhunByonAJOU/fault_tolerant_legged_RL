# 논문 LaTeX 편집

## 2026-10-16 최종 데이터 및 Windows 인계

새 평가는 완료됐다. **2:1:1:2 구성의 26,352개 시행과 모든 50Hz 원시 기록**은 저장소 루트의 [datasets/ieie2026_fixed2112](../../../../../datasets/ieie2026_fixed2112/README.md)에 있다. Git LFS 다운로드와 Windows 분석 명령은 해당 README를 따른다. 최신 인계 기준은 [HANDOFF.txt](HANDOFF.txt)다.

생존율은 **낙상 여부의 평균**이며 정규화 평균 생존 시간이 아니다. 현재 원고의 실험 결과 문단·기존 그림은 아직 예전20초 직진 평가이므로 새 데이터의 표·그림으로 교체해야 한다. 아래의 “추가 평가 미실행” 등은 이전 이관 시점의 기록이다. 서론·초록 문장·용어·인용 공백 수정은 반영했다. 프로시딩6쪽/심사용5쪽을 각각5쪽/4쪽으로 줄이는 작업은 남아 있다.

## Ubuntu에서 이어서 작업하기 (2026-10-16)

저장소 **루트**의 [ieie-paper.code-workspace](../../../../../ieie-paper.code-workspace)를 열면 Git 저장소와 논문 빌드 설정을 함께 사용한다. `capstone_ws` 자체는 이 GitHub 저장소가 아니며 실제 저장소는 그 아래 `ieie-paper`에 있다. 기존 Ubuntu 체크아웃에서는 `git switch dev`, `git pull --ff-only` 후 루트 작업 영역을 연다.

Ubuntu에 빌드 도구가 없다면 다음 패키지를 설치한다.

```sh
sudo apt update
sudo apt install latexmk texlive-luatex texlive-lang-korean texlive-latex-extra texlive-fonts-recommended fonts-noto-cjk
```

저장소 루트에서 빌드하려면:

```sh
cd docs/submission/ieie2026/private/latex
latexmk -lualatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build proceedings.tex
```

Windows 양식 글꼴이 없는 Ubuntu에서는 대체 글꼴로 편집한다. 현재 배치의 비교 기준은 `preview/proceedings.pdf`이며, 최종 양식 확인은 동일한 글꼴 환경에서 다시 해야 한다. 글꼴 바이너리는 Git에 포함하지 않는다. 이 Windows PC에는 실행 가능한 WSL 배포판이 없어 Ubuntu 자체 빌드까지 검증한 것은 아니다.

2026-10-16 재빌드에서 프로시딩은 **6쪽**이었다. 재빌드 전의 기존 프리뷰도 6쪽으로, 이번 이식 설정 변경으로 늘어난 것은 아니다. 아래 2026-10-12의 5쪽 확인 기록과 구별해야 한다. 현재 원고는 **5쪽 한도에 맞춘 추가 축약이 필요**하며, 이번 이관에서는 본문과 결과를 임의로 줄이지 않았다.

추가 평가에 대한 합의는 [HANDOFF.txt](HANDOFF.txt)에 정리했다. **새 평가 지표·과제는 아직 원고의 실험 결과로 반영하지 않았다.** 현재 본문에는 기존 직진 평가의 종료 시 생존율·변위·Q가 남아 있다. 새 데이터를 받은 뒤 교체한다.

제출 한도는 프로시딩 **5쪽 이내**, 익명 서면 심사용 **4쪽**이다. 2026-10-12 Windows 빌드 기준, 프로시딩은 **5쪽**, 기존 심사용은 **5쪽**이다. 프로시딩은 참고 논문 NODE12551200의 제목부 배치와 이메일 표기를 반영하고, 반복 설명과 관측·인코더 표를 본문으로 압축했다. 결과 수치와 그림 5개는 유지했다. 심사용은 이후 4쪽으로 축약해야 한다.

| 용도 | 직접 편집하는 본문 | 빌드 진입 파일 | 최신 PDF |
|---|---|---|---|
| 프로시딩 | [body.tex](body.tex) | [proceedings.tex](proceedings.tex) | [build/proceedings.pdf](build/proceedings.pdf) |
| 서면 심사용 | [body_review.tex](body_review.tex) | [main.tex](main.tex) | [build/main.pdf](build/main.pdf) |

`preview/proceedings.pdf`와 `preview/main.pdf`는 2026-10-16 재빌드 결과이며 각각 6쪽과 5쪽이다. 두 원고 모두 제출 한도에 맞춘 추가 축약이 필요하다. 직접 빌드한 결과는 Git에서 제외되는 `build/`에 생성된다.

이 Windows PC에서는 저장소 바깥의 `capstone_ws/ieie-paper-windows.code-workspace`를 연다. `latex-build-windows.ps1`을 통해 LuaLaTeX를 실행한다. XeLaTeX의 PDF 변환기 충돌을 피하도록 구성되어 있으며, Ctrl+Alt+B 빌드와 Ctrl+Alt+V 미리보기를 사용한다.

## 노트북에서 열기

1. 저장소의 `dev` 브랜치를 받는다. 이미 저장소가 있으면 `git switch dev` 후 `git pull --ff-only`를 실행한다.
2. VS Code의 **파일 → 파일에서 작업 영역 열기**로 [ieie-paper.code-workspace](ieie-paper.code-workspace)를 연다.
3. 권장 확장 **LaTeX Workshop**을 설치한다. LuaLaTeX와 latexmk를 포함하는 TeX Live 또는 MiKTeX가 PATH에서 실행되어야 한다. 프로시딩의 `paperdisplay` 처리가 LuaTeX를 사용하므로 XeLaTeX로 빌드하지 않는다.
4. [body.tex](body.tex)를 편집한다. **Ctrl+S**로 저장하면 프로시딩을 자동 빌드한다. **Ctrl+Alt+B**는 수동 빌드, **Ctrl+Alt+V**는 옆 탭 프리뷰다.
5. 심사용을 편집하려면 [body_review.tex](body_review.tex)를 연다. 파일 첫 줄의 root 주석이 각각 proceedings.tex/main.tex를 선택한다.

명령줄 빌드는 이 디렉터리에서 실행한다.

```sh
latexmk -lualatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build proceedings.tex
latexmk -lualatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build main.tex
```

## 글꼴과 공통 파일

- [fonts-proceedings.tex](fonts-proceedings.tex): 프로시딩 DOCX에서 읽은 글꼴을 지정한다. Windows의 본문·초록·캡션·참고문헌은 HY신명조, 국문 제목은 바탕, 국문 저자명은 한컴바탕, 수식은 Cambria Math를 사용한다. 다른 OS에서는 설치된 동일 글꼴을 우선하고, 없으면 [fonts.tex](fonts.tex)의 편집용 글꼴과 Latin Modern Math로 대체한다. 글꼴이 다르면 줄바꿈과 쪽 수가 달라질 수 있다. 그래프 PDF의 글꼴은 이미 포함되어 있다. 폰트 파일은 저장소에 포함하지 않는다.
- [fonts.tex](fonts.tex): 심사용 원고의 공통 글꼴 선택과 대체 규칙이다. 프로시딩은 위의 전용 설정을 사용한다.
- [frontmatter.tex](frontmatter.tex): 제목·한영 초록. 두 원고에서 공유한다.
- [authors.tex](authors.tex): 프로시딩의 저자·소속. 심사용에는 포함하지 않는다.
- [frontmatter-proceedings.tex](frontmatter-proceedings.tex): DOCX 순서에 맞춘 한영 제목·저자·소속과 2단 English Abstract. 프로시딩에는 양식에 없는 국문 요약과 Keywords 블록을 출력하지 않는다.
- [format-proceedings.tex](format-proceedings.tex): 프로시딩의 본문·제목·표·캡션·참고문헌 서식. 본문 HY신명조 9 Word pt, 실제 Word 출력 기준 줄 간격 14.04 pt. 한글 제목 바탕 18 pt 굵게, 장평 95%; 영문 제목 12 pt 굵게. 본문 단락 간격 0, 들여쓰기 6 pt. 위아래 여백 약 30 mm, 좌우 약 20 mm, 단 간격 약 10 mm.
- `fonts-local.tex`: 심사용 원고의 이 PC 전용 설정이며 Git에서 제외한다. 프로시딩은 이전 `fonts-proceedings-local.tex` 대신 Git 관리 대상인 `fonts-proceedings.tex`를 직접 읽는다. 본문에 없는 일부 기호에만 양식의 fontTable에도 포함된 Times New Roman을 사용한다.
- [preamble.tex](preamble.tex): 공통 설정과 양식별 분기. 심사용의 기존 설정은 유지한다.
- [../assets/](../assets/): 두 본문에서 사용하는 이미지 6개만 보관한다. 히트맵은 Q20 하나이다. 파일명의 버전 번호는 선택된 그림의 출처이며, 다른 버전의 그림은 제거했다.

공식 양식은 DOCX/HWP이며 이 TeX는 공식 LaTeX 양식이 아니다. 프로시딩은 공식 `논문양식.docx`의 XML과 이 PC의 Word PDF 출력에서 측정한 서식을 적용했다. Word와 TeX의 줄바꿈·수식 배치는 완전히 같지 않다. 샘플에 없는 수식·데이터 표의 구조는 기존 것을 유지했다. [서식 근거](../../sources/proceedings_template_format.json)에 출처와 측정값을 기록한다. `body.tex`와 `body_review.tex`는 별도 본문이므로 공통 문장의 수정은 두 파일에 반영한다.

새 학습·시뮬레이션 없이 기존 평가와 학습 로그로 작성했다. 이전 원고·백업·Word 초안·중간 렌더링은 정리했고, 이후 빌드 캐시는 계속 Git에서 제외한다.

2026-10-12 메일 및 구두 수정 사항을 프로시딩에 반영했다. 5쪽 기준을 유지하며 그림·표는 단 상단에 배치하고, 본문에서 번호로 참조한다. 인용은 문장 끝에 두며 참고문헌의 DOI와 `pp.` 표기는 제거했다. 장·절 제목은 한글까지 굵게 표시하고, 수식에는 설명·번호 참조·문장부호와 위아래 간격을 적용했다. 연속된 짧은 결과 문단과 결론을 묶고 필요한 약어를 풀었다. 첫 쪽 Abstract 제목과 오른쪽 단 본문의 윗선을 기존 두 높이의 중간에 맞췄으며, 그림 1의 소그림 설명과 전체 캡션 간격은 3 bp로 줄였다. 심사용 본문은 이번 프로시딩 수정 대상에 포함하지 않았다.

후속 수정에서는 모든 표의 제목과 표 사이 간격, 그림 3의 그래프와 캡션 사이 간격을 0 bp로 설정했다. 수식 묶음은 `paperdisplay`로 감싸 위아래 고정 간격 14.04 bp를 적용하고, 묶음 내부의 추가 행 간격은 0 bp이다. 영문 저자명은 `Eunbin Yoo`, `CHOI SEOKYEONG`으로 수정했다. 그래프 PDF 3개와 구조도 PDF의 실제 글자는 모두 HY신명조로 교체하고, 수치·도형 경로·색상 데이터는 유지했다. 재현용 [글꼴 교체 스크립트](../../restyle_figure_text.py)는 기존 벡터 PDF와 설치된 폰트를 입력받는다. 본문·초록·표·그림 및 심사용 사본까지 숫자와 퍼센트·SI 단위 사이를 띄웠다. 할인율은 `할인 계수(discount factor)`로 명확히 쓰고 학습률과 구분했다. 그림 3·4는 4쪽의 좌우 단 상단에 나눠 배치했다. Isaac Gym은 공식 NeurIPS Datasets and Benchmarks 2021 출판 기록으로 인용하고, Saving the Limping은 arXiv v3에서 RA-L 투고만 확인되어 해당 버전을 유지한다.

2026-10-12 문장 개정에서는 프로시딩의 영문 초록과 본문 전체를 다듬었다. 고장 이후의 생존과 명령 수행을 서론에서 구분하고, 관측 이력의 목적을 차원 설명보다 먼저 제시했다. 방법은 관측 → 공동학습 → 보상 → 학습 설정 순서로 정리했다. 결과 문단은 수치 반복을 줄이고 강한 고장의 이점, 낮은 손실률의 이동량 감소, calf 취약성을 해석한다. `Normal`과 `Student`는 그림과 대응하도록 각각 정상학습 정책과 고장대응 정책으로 정의했다. [독자 관점 검토·제공된 논문 5편 분석·공개 비교 코드 조사](../../reviewer/reader_review_20261012.txt)를 별도로 남겼다. 공개 코드 조사 결과는 원고에 추가하지 않았다. 이번 문장 개정은 프로시딩에 적용했으며 심사용 본문은 이후 4쪽 축약 시 반영한다.
