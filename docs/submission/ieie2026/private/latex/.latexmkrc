# Use XeLaTeX for this Korean manuscript, also from a terminal.
$pdf_mode = 5;
$out_dir = 'build';
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';
