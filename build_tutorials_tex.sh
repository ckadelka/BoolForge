#!/usr/bin/env bash
set -euo pipefail

# Toggle arXiv build (default: false)
BUILD_ARXIV=${1:-false}

# ------------------------------------------------------------
# Build a single LaTeX file from BoolForge tutorial notebooks
# ------------------------------------------------------------

TUTORIAL_DIR="tutorials"
MD_DIR="tutorials_md"
OUT_TEX="BoolForge_Tutorials.tex"
HEADER_TEX="tutorials_header.tex"

TITLE="Boolean Network Modeling in Systems Biology: A Hands-On Tutorial with BoolForge"
AUTHOR="Claus Kadelka"

echo "==> Cleaning previous build artifacts"
rm -rf "$MD_DIR" "$OUT_TEX" "$HEADER_TEX"
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
# Create LaTeX header (arXiv-safe)
# ------------------------------------------------------------
echo "==> Writing LaTeX header"

cat > "$HEADER_TEX" <<'EOF'
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{listings}
\usepackage{hyperref}

% allow alt= in includegraphics
\makeatletter
\define@key{Gin}{alt}{}
\makeatother

% prevent oversized figures
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\makeatother

\setkeys{Gin}{width=\maxwidth,height=0.8\textheight,keepaspectratio}

% code formatting
\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single
}

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
# Build LaTeX
# ------------------------------------------------------------
  
if [ "$BUILD_ARXIV" = true ]; then
    echo "==> Building arXiv version with abstract + intro"
    cat arxiv_frontmatter.md "$MD_DIR"/*.md > "$MD_DIR/_combined.md"
    
    pandoc \
      "$MD_DIR/_combined.md" \
      --resource-path="$MD_DIR" \
      --from markdown-yaml_metadata_block+tex_math_dollars \
      --standalone \
      --toc \
      --number-sections \
      --listings \
      -V geometry:margin=1in \
      -V documentclass=article \
      -H "$HEADER_TEX" \
      -o "$OUT_TEX"

else
    echo "==> Building supplement version (no abstract)"
    
    pandoc \
      "$MD_DIR"/*.md \
      --resource-path="$MD_DIR" \
      --from markdown-yaml_metadata_block+tex_math_dollars \
      --standalone \
      --toc \
      --number-sections \
      --listings \
      -M title="$TITLE" \
      -M author="$AUTHOR" \
      -M date="$(date '+%B %d, %Y')" \
      -V geometry:margin=1in \
      -V documentclass=article \
      -H "$HEADER_TEX" \
      -o "$OUT_TEX"
fi

echo "==> Done: $OUT_TEX"