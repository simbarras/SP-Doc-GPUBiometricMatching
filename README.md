# Semester Project - GPU Biometric Matching

Latex sources of the documentation for the semester project "GPU Biometric Matching" at EPFL.


## Compile doc with latexmk
To compile the doc manually with latexmk, the command will automatically use the file `.latexmkrc`:
```shell
latexmk --pdf report.tex
```

## Compile doc with pdlatex
To compile the doc manually, uses this:
``` shell
pdflatex report.tex
makeglossaries report
pdflatex report.tex
```