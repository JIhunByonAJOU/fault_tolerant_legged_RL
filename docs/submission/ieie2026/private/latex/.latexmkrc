# Proceedings formatting uses LuaTeX node processing; use LuaLaTeX on all OSes.
$pdf_mode = 4;
$out_dir = 'build';
$lualatex = 'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';
