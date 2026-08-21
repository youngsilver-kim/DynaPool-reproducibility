# Colab to a new GitHub repository

Recommended repository name: `dynapool-reproducibility`.

## 1. Open the clean project in Colab

Upload `dynapool-reproducibility.zip` to the Colab Files panel, then run:

```python
!unzip -q /content/dynapool-reproducibility.zip -d /content
%cd /content/dynapool-reproducibility
```

## 2. Verify before committing

```python
!python -m pip install -q -r requirements.txt
!python -m pytest -q
!python scripts/plot_figure3.py
```

The Figure 3 command creates a 600-dpi PNG, a vector PDF, and a statistics CSV
under `paper_figures/`. That directory is ignored by Git because these outputs
are reproducible; attach the PNG or PDF to the manuscript submission instead.

## 3. Create the local commit

Replace the name and email with your own GitHub commit identity:

```python
!git init
!git config user.name "YOUR NAME"
!git config user.email "YOUR VERIFIED GITHUB EMAIL"
!git add .
!git status --short
!git commit -m "Release reproducible DynaPool experiments and Figure 3"
!git branch -M main
```

## 4. Create the new GitHub repository and push

Install GitHub CLI, authenticate with the browser device flow, and replace
`YOUR_GITHUB_ID` below:

```python
!apt-get -qq update
!apt-get -qq install gh
!gh auth login --web --git-protocol https
!gh repo create YOUR_GITHUB_ID/dynapool-reproducibility \
    --private \
    --description "Reproducible Tiny-ImageNet experiments and analysis for DynaPool" \
    --source=. \
    --remote=origin \
    --push
```

Use `--public` instead of `--private` only when the repository is ready to be
shared. `git commit` saves to the temporary Colab VM; `--push` is the step that
actually transfers the commit to GitHub.

## 5. Confirm the remote state

```python
!git status
!git remote -v
!git log --oneline -1
!gh repo view YOUR_GITHUB_ID/dynapool-reproducibility --web
```

Never paste a personal access token into a visible notebook cell or commit it
to the repository.
