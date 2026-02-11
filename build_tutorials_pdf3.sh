#!/usr/bin/env bash
set -e

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
TUTORIAL_DIR="tutorials"
MD_DIR="tutorials_md"
OUT_MD="BoolForge_Tutorials.md"
OUT_PDF="BoolForge_Tutorials.pdf"
HEADER_TEX="tutorials_header.tex"

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
echo "==> Cleaning previous build artifacts"
rm -rf "$MD_DIR" "$OUT_MD" "$OUT_PDF" "$HEADER_TEX"
mkdir -p "$MD_DIR"

# ------------------------------------------------------------
# Convert notebooks to Markdown
# ------------------------------------------------------------
echo "==> Converting notebooks to Markdown"
for nb in "$TUTORIAL_DIR"/*.ipynb; do
  jupyter nbconvert \
    --to markdown \
    "$nb" \
    --output-dir "$MD_DIR"
done

# ------------------------------------------------------------
# Strip YAML front matter (TOP OF FILE ONLY)
# ------------------------------------------------------------
echo "==> Stripping YAML front matter"
for md in "$MD_DIR"/*.md; do
  if head -n 1 "$md" | grep -q '^---$'; then
    sed -i '' '1,/^---$/d' "$md"
  fi
done

# ------------------------------------------------------------
# LaTeX header
# ------------------------------------------------------------
echo "==> Writing LaTeX header"
cat > "$HEADER_TEX" << 'EOF'
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{margin=1in}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}

\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{keyval}

% ---- Ignore Pandoc's alt=... image key ----
\makeatletter
\define@key{Gin}{alt}{}
\makeatother
EOF

# ------------------------------------------------------------
# Concatenate tutorials
#   - force first heading of each file to be chapter (#)
# ------------------------------------------------------------
echo "==> Concatenating Markdown files"
> "$OUT_MD"

for md in "$MD_DIR"/*.md; do
    echo "" >> "$OUT_MD"
    echo "\\pagebreak" >> "$OUT_MD"
    echo "" >> "$OUT_MD"
  awk '
    BEGIN { first = 1 }
    {
      if ($0 ~ /^#+[[:space:]]+/ && first) {
        sub(/^#+[[:space:]]+/, "# ")
        first = 0
      }
      print
    }
  ' "$md" >> "$OUT_MD"
done

# ------------------------------------------------------------
# Build PDF
# ------------------------------------------------------------
echo "==> Building final PDF"
pandoc "$OUT_MD" \
  -f markdown-yaml_metadata_block \
  -o "$OUT_PDF" \
  --pdf-engine=xelatex \
  --toc \
  --number-sections \
  -H "$HEADER_TEX" \
  --resource-path=.:tutorials_md

echo "==> Done!"
echo "Generated: $OUT_PDF"
