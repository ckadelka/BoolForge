#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Build a single PDF from BoolForge tutorial notebooks
# ------------------------------------------------------------

TUTORIAL_DIR="tutorials"
MD_DIR="tutorials_md"
OUT_PDF="BoolForge_Tutorials.pdf"
HEADER_TEX="tutorials_header.tex"

echo "==> Cleaning previous build artifacts"
rm -rf "$MD_DIR" "$OUT_PDF" "$HEADER_TEX"
mkdir -p "$MD_DIR"

# ------------------------------------------------------------
# Convert notebooks to Markdown
# ------------------------------------------------------------
echo "==> Converting notebooks to Markdown"

for nb in "$TUTORIAL_DIR"/*.ipynb; do
  jupyter nbconvert \
    --to markdown \
    --TemplateExporter.exclude_input_prompt=True \
    --TemplateExporter.exclude_output_prompt=True \
    "$nb" \
    --output-dir "$MD_DIR"
done

# ------------------------------------------------------------
# Remove "png" captions
# ------------------------------------------------------------
echo "==> Cleaning figure captions"

for f in "$MD_DIR"/*.md; do
    sed -i '' 's/!\[png\]/![]/g' "$f"
done

# ------------------------------------------------------------
# Add tutorial titles from filenames
# ------------------------------------------------------------
#echo "==> Adding tutorial titles"
#
#for f in "$MD_DIR"/*.md; do
#    title=$(basename "$f" .md | sed 's/_/ /g')
#    sed -i '' "1s/^/# $title\n\n/" "$f"
#done

# ------------------------------------------------------------
# Create LaTeX header
# ------------------------------------------------------------
echo "==> Writing LaTeX header"

cat > "$HEADER_TEX" <<'EOF'
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}

% allow alt= in includegraphics
\makeatletter
\define@key{Gin}{alt}{}
\makeatother

% prevent oversized figures
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\makeatother

\setkeys{Gin}{width=\maxwidth,height=0.8\textheight,keepaspectratio}

% new page per tutorial
\usepackage{etoolbox}
\pretocmd{\section}{\clearpage}{}{}

% nicer captions
\captionsetup{
  font=small,
  labelfont=bf
}
EOF

# ------------------------------------------------------------
# Build PDF
# ------------------------------------------------------------
echo "==> Building $OUT_PDF"

pandoc \
  "$MD_DIR"/*.md \
  --resource-path="$MD_DIR" \
  --from markdown-yaml_metadata_block+tex_math_dollars \
  --pdf-engine=xelatex \
  --toc \
  --number-sections \
  --syntax-highlighting=tango \
  -V geometry:margin=1in \
  -V documentclass=article \
  -V mainfont="Latin Modern Roman" \
  -V monofont="Menlo" \
  -V colorlinks=true \
  -V linkcolor=blue \
  -H "$HEADER_TEX" \
  -o "$OUT_PDF"

echo "==> Done: $OUT_PDF"