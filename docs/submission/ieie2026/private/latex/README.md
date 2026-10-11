# 논문 LaTeX 편집

현재 프로시딩은 **5쪽**, 익명 서면 심사용은 **4쪽**이다. 프로시딩을 먼저 수정한 뒤 심사용에 내용을 축약해 반영한다.

| 용도 | 직접 편집하는 본문 | 빌드 진입 파일 | 최신 PDF |
|---|---|---|---|
| 프로시딩 5쪽 | [body.tex](body.tex) | [proceedings.tex](proceedings.tex) | [preview/proceedings.pdf](preview/proceedings.pdf) |
| 서면 심사용 4쪽 | [body_review.tex](body_review.tex) | [main.tex](main.tex) | [preview/main.pdf](preview/main.pdf) |

`preview/`는 커밋 시점의 최신 PDF 두 개이다. 직접 빌드한 결과는 Git에서 제외되는 `build/`에 생성된다.

## 노트북에서 열기

1. 저장소의 `dev` 브랜치를 받는다. 이미 저장소가 있으면 `git switch dev` 후 `git pull --ff-only`를 실행한다.
2. VS Code의 **파일 → 파일에서 작업 영역 열기**로 [ieie-paper.code-workspace](ieie-paper.code-workspace)를 연다.
3. 권장 확장 **LaTeX Workshop**을 설치한다. XeLaTeX와 latexmk를 포함하는 TeX Live 또는 MiKTeX가 PATH에서 실행되어야 한다.
4. [body.tex](body.tex)를 편집한다. **Ctrl+S**로 저장하면 프로시딩을 자동 빌드한다. **Ctrl+Alt+B**는 수동 빌드, **Ctrl+Alt+V**는 옆 탭 프리뷰다.
5. 심사용을 편집하려면 [body_review.tex](body_review.tex)를 연다. 파일 첫 줄의 root 주석이 각각 proceedings.tex/main.tex를 선택한다.

명령줄 빌드는 이 디렉터리에서 실행한다.

```sh
latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build proceedings.tex
latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build main.tex
```

## 글꼴과 공통 파일

- [fonts.tex](fonts.tex): 설치된 글꼴 이름으로 선택한다. 사용자 홈 디렉터리나 Linux 전용 경로를 사용하지 않는다. KoPubWorld 바탕 Medium/Bold와 돋움 Medium/Bold를 설치하면 원고의 글꼴을 재현한다. 미설치 시 Noto CJK, 은글꼴 또는 Windows 한글 글꼴을 사용한다. 대체 글꼴에서는 줄바꿈과 쪽수가 달라질 수 있다. 글꼴 파일은 저장소에 포함하지 않는다.
- [frontmatter.tex](frontmatter.tex): 제목·한영 초록. 두 원고에서 공유한다.
- [authors.tex](authors.tex): 프로시딩의 저자·소속. 심사용에는 포함하지 않는다.
- [preamble.tex](preamble.tex): 양식별 본문·제목·표·캡션 간격. 프로시딩 본문은 9pt/10.8pt, 심사용은 10pt/16pt이다.
- [../assets/](../assets/): 두 본문에서 사용하는 이미지 6개만 보관한다. 히트맵은 Q20 하나이다. 파일명의 버전 번호는 선택된 그림의 출처이며, 다른 버전의 그림은 제거했다.

공식 양식은 DOCX/HWP이며 이 TeX는 두 양식의 여백·단 간격·글자 크기·줄 간격을 참고한 편집본이다. `body.tex`와 `body_review.tex`는 별도 본문이므로 공통 문장의 수정은 두 파일에 반영한다.

새 학습·시뮬레이션 없이 기존 평가와 학습 로그로 작성했다. 이전 원고·백업·Word 초안·중간 렌더링은 정리했고, 이후 빌드 캐시는 계속 Git에서 제외한다.
