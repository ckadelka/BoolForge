#!/usr/bin/env bash
set -euo pipefail

# Toggle arXiv build (default: false)
BUILD_ARXIV=$(echo "${1:-false}" | tr '[:upper:]' '[:lower:]')

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

TUTORIAL_DIR="tutorials"
MD_DIR="tutorials_md"
OUT_TEX="BoolForge_Tutorials.tex"
HEADER_TEX="tutorials_header.tex"
FIG_DIR="figures"

TITLE="Boolean Network Modeling in Systems Biology: A Hands-On Tutorial with BoolForge"
AUTHOR="Claus Kadelka"

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------

echo "==> Cleaning previous build artifacts"
rm -rf "$MD_DIR" "$OUT_TEX" "$HEADER_TEX" "$FIG_DIR"
mkdir -p "$MD_DIR" "$FIG_DIR"

# ------------------------------------------------------------
# Convert notebooks → Markdown
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
# Clean captions (portable sed)
# ------------------------------------------------------------

echo "==> Cleaning figure captions"

for f in "$MD_DIR"/*.md; do
    sed -i.bak 's/!\[png\]/![]/g' "$f"
    rm "$f.bak"
done


# ------------------------------------------------------------
# FIX 1: ensure headings start new lines
# ------------------------------------------------------------

echo "==> Fixing missing blank lines before headings"

for f in "$MD_DIR"/*.md; do
    sed -i.bak 's/^\(# .*\)/\n\1/g' "$f"
    rm "$f.bak"
done

# ------------------------------------------------------------
# FIX 2: remove accidental bold formatting
# ------------------------------------------------------------

echo "==> Removing accidental bold formatting"

for f in "$MD_DIR"/*.md; do
    sed -i.bak 's/\*\*/ /g' "$f"
    rm "$f.bak"
done

# ------------------------------------------------------------
# FIX 3: replace Unicode logical symbols 
# ------------------------------------------------------------

echo "==> Replacing Unicode logical symbols (∧, ∨, ¬)"

for f in "$MD_DIR"/*.md; do
    sed -i.bak 's/∧/\\wedge/g' "$f"
    sed -i.bak 's/∨/\\vee/g' "$f"
    sed -i.bak 's/¬/\\neg/g' "$f"
    
    rm "$f.bak"
done


# ------------------------------------------------------------
# Flatten figures → figures/
# ------------------------------------------------------------

echo "==> Flattening figure directories"

for md in "$MD_DIR"/*.md; do
    base=$(basename "$md" .md)
    files_dir="$MD_DIR/${base}_files"

    if [ -d "$files_dir" ]; then
        counter=0

        for img in "$files_dir"/*; do
            ext="${img##*.}"
            new_name="${base}_fig${counter}.${ext}"

            cp "$img" "$FIG_DIR/$new_name"

            sed -i.bak "s|${base}_files/$(basename "$img")|${FIG_DIR}/${new_name}|g" "$md"
            rm "$md.bak"

            ((counter++))
        done
    fi
done

rm -rf "$MD_DIR"/*_files

# ------------------------------------------------------------
# LaTeX header (FIXES ALL YOUR ISSUES)
# ------------------------------------------------------------

echo "==> Writing LaTeX header"

cat > "$HEADER_TEX" <<'EOF'
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{url}
\usepackage{etoolbox}
\usepackage{natbib}

\renewcommand{\refname}{}
\renewcommand{\bibname}{}

% --- Title handling ---
\title{$title$}
\author{$author$}
\date{$date$}

\AtBeginDocument{
  \maketitle
  %\tableofcontents
  \clearpage
}

% --- allow alt= ---
\makeatletter
\define@key{Gin}{alt}{}
\makeatother

% --- scale figures ---
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\makeatother

\setkeys{Gin}{width=\maxwidth,height=0.8\textheight,keepaspectratio}

% --- syntax highlighting (arXiv-safe) ---
\lstdefinelanguage{Python}{
  keywords={def, return, if, else, elif, for, while, import, from, as, class, try, except, raise, with, lambda, yield, True, False, None},
  keywordstyle=\color{blue}\bfseries,
  comment=[l]{\#},
  commentstyle=\color{gray}\itshape,
  stringstyle=\color{red},
  morestring=[b]',
  morestring=[b]",
}

\lstset{
  language=Python,
  basicstyle=\ttfamily\small,
  keywordstyle=\color{blue}\bfseries,
  commentstyle=\color{gray}\itshape,
  stringstyle=\color{red},
  breaklines=true,
  frame=single,
  showstringspaces=false,
  columns=fullflexible,
  keepspaces=true
}

% --- new page per section ---
%\pretocmd{\section}{\clearpage}{}{}
\makeatletter
\pretocmd{\section}{%
  \ifnum\pdfstrcmp{\@currentlabelname}{References}=0
    % do nothing
  \else
    \clearpage
  \fi
}{}{}
\makeatother

% --- nicer captions ---
\captionsetup{
  font=small,
  labelfont=bf
}

% --- FIX bold-text bug ---
\normalfont
\renewcommand{\familydefault}{\rmdefault}
EOF

# ------------------------------------------------------------
# Build LaTeX
# ------------------------------------------------------------

echo "==> Building LaTeX"

if [ "$BUILD_ARXIV" = true ]; then
    echo "==> arXiv version (with abstract + intro)"

    files=()
    for f in "$MD_DIR"/tutorial*.md; do
        if [[ "$f" != *tutorial00*.md && "$f" != *tutorial12*.md ]]; then
            files+=("$f")
        fi
    done


    {
      cat arxiv_abstract.md
      printf "\n\n"
    
      cat "$MD_DIR"/tutorial00_preface_tex.md
      printf "\n\n"
    
      cat arxiv_frontmatter.md
      printf "\n\n"
    
      for f in "${files[@]}"; do
        cat "$f"
        printf "\n\n"
      done

      cat arxiv_backmatter.md
    } > "$MD_DIR/_combined.md"

    pandoc \
      "$MD_DIR/_combined.md" \
      --resource-path="$MD_DIR" \
      --from markdown+yaml_metadata_block+tex_math_dollars+fenced_code_blocks \
      --toc \
      --standalone \
      --number-sections \
      --listings \
      --syntax-highlighting=none \
      --natbib --bibliography=references.bib \
      -V title="$TITLE" \
      -V author="$AUTHOR" \
      -V date="$(date '+%B %d, %Y')" \
      -V geometry:margin=1in \
      -V documentclass=article \
      -H "$HEADER_TEX" \
      -o "$OUT_TEX"
      
    echo "==> Converting verbatim to lstlisting (force syntax highlighting)"
    
    sed -i.bak 's/\\begin{verbatim}/\\begin{lstlisting}/g' "$OUT_TEX"
    sed -i.bak 's/\\end{verbatim}/\\end{lstlisting}/g' "$OUT_TEX"
    rm "$OUT_TEX.bak"

else
    echo "==> supplement version"

    files=()
    for f in "$MD_DIR"/tutorial*.md; do
        if [[ "$f" != *tutorial00_preface_tex.md ]]; then
            files+=("$f")
        fi
    done

    pandoc \
      "${files[@]}" \
      --resource-path="$MD_DIR" \
      --from markdown+yaml_metadata_block+tex_math_dollars \
      --standalone \
      --number-sections \
      --listings \
      --syntax-highlighting=none \
      -V title="$TITLE" \
      -V author="$AUTHOR" \
      -V date="$(date '+%B %d, %Y')" \
      -V geometry:margin=1in \
      -V documentclass=article \
      -H "$HEADER_TEX" \
      -o "$OUT_TEX"
fi

echo "==> Done: $OUT_TEX"