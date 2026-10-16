# 편집기 PDF의 참조·인용 오류 수정

문제가 발생한 파일은 `private/latex/build/proceedings.pdf`였다. 저장소 루트 `.vscode/settings.json`이 XeLaTeX을 선택한 반면 논문·워크스페이스는 LuaLaTeX을 사용했다. 편집기 출력의 미해결 교차참조 24개와 문헌 인용 11개를 확인했다. 별도 검증·배포 PDF만 정상임을 확인하고 편집기 출력을 함께 검증하지 않은 누락이었다. 인용 키를 삭제하거나 번호를 수동으로 적어 해결한 것이 아니다.

## 수정

- 저장소 루트·논문 폴더의 VS Code 설정 두 개와 workspace 두 개를 모두 `latexmk -lualatex`, 명시적 LuaLaTeX recipe, 저장 시 자동 빌드로 통일했다.
- 두 진입 TeX에 `\RequireLuaTeX`를 추가하여 잘못된 엔진을 쓰면 조판 전에 명확히 중단한다.
- 실제 편집기 출력 `build/`의 두 원고를 완전 정리한 뒤 LuaLaTeX 다중 패스로 다시 빌드했다.
- 검증이 전부 통과한 canonical build만 `preview/`로 복사했다. 두 경로의 PDF는 바이트 단위로 같다.

## 전체 검사 결과

두 원고 각각 그림 6개, 표 6개, 수식 9개와 참고문헌 6개를 확인했다. 번호의 정의·사용·대상 페이지를 대조하고 PDF의 실제 클릭 링크도 확인했다. 미해결 참조·인용, 중복 라벨, 없는 PDF 목적지와 문헌 목적지, 빠진 글리프, 넘친 상자는 없다. 첫 인용 순서·문장 끝 붙임 인용·익명성을 검사했다. 기존 수치·낙상률 60셀·과제별 생존/완주 정수 검산도 통과했다. 원자료·과학적 주장·그림 구성은 바꾸지 않았다.

프로시딩의 내부 PDF 링크는 34개, 익명본은 글꼴 구간별 분할을 포함해 74개다. 두 경우 모두 원고의 21개 참조 라벨과 여섯 문헌의 클릭 대상 페이지를 확인했다. 링크 수는 서로 다른 글꼴 구간으로 링크가 나뉘는 방식 때문에 다르다.

현재 두 원고는 6쪽이며 목표 5쪽·4쪽 한도는 여전히 미충족이다. 실제 VS Code UI에서 저장 키를 누르는 시험은 하지 않았으며, 네 설정 파일을 대조하고 그 설정과 같은 명령으로 `build/`를 새로 생성했다.

[수정 전후 감사 기록](paper_assets/reference_audit_20261017.json), [전체 검사](paper_assets/manuscript_validation.json)을 제공한다. 기존 오류 PDF를 새 검사에 넣으면 거부되고, XeLaTeX을 임시 출력 경로에서 시도하면 LuaTeX 필요 메시지로 조판 전에 중단함을 확인했다.

```sh
cd docs/submission/ieie2026/private/latex
latexmk -lualatex -outdir=build proceedings.tex
latexmk -lualatex -outdir=build main.tex
cd ../../../../..
python3 docs/submission/ieie2026/validate_manuscript.py --publish-previews
```

검사기는 `pdftotext`, `pdftohtml`, `pdfinfo`, `pdffonts`를 사용한다. 원고 편집 시에는 `build/` PDF를 보고, 배포할 때 위 검증을 통과한 `preview/` 사본을 사용한다. PDF를 한 번만 컴파일하거나 서로 다른 엔진으로 같은 출력 경로를 덮어쓰지 않는다.
