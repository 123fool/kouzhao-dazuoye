#!/usr/bin/env perl
# Latexmk configuration file to use xelatex for compiling Chinese documents

$pdf_mode = 5;  # 5 means use xelatex
$postscript_mode = 0;
$dvi_mode = 0;

# Use xelatex
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Other useful settings
$max_repeat = 5;
$out_dir = '.';
