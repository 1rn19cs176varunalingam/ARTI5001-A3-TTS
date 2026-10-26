# Git from the terminal: a hands-on course using this project

Type every command yourself; don't copy-paste blocks blindly. After each command, run `git status`
and read what it says. That habit alone prevents most git mistakes.

**Mental model:** git has three places your changes move through:
```
working directory --git add--> staging area (index) --git commit--> repository (history) --git push--> GitHub
      (your files)               (what goes in the next commit)       (local snapshots)                 (shared)
```

---

## Lesson 0: One-time setup (per computer)
```bash
git config --global user.name  "Your Name"          # shown on every commit
git config --global user.email "you@example.com"    # use the email on your GitHub account
git config --global init.defaultBranch main
git config --global pull.rebase false               # 'git pull' merges (simplest for beginners)
git config --global core.editor "code --wait"       # VS Code for commit messages (or nano)
git config --global --list                          # check it all
```
GitHub login from the terminal (easiest way):
```bash
brew install gh
gh auth login        # GitHub.com → HTTPS → "Login with a web browser"
```

## Lesson 1: Create the repo and make the first commit (Varun)
```bash
cd ~/ARTI5001-A3-TTS
git init                       # creates the hidden .git/ folder: this is now a repository
git status                     # everything is "untracked"; note that data/ is NOT listed (.gitignore)
git check-ignore -v data/LJSpeech-1.1.tar.bz2   # proves which .gitignore rule hides the dataset
git add .gitignore             # stage one file
git status                     # .gitignore is now green, "Changes to be committed"
git add .                      # stage everything else that isn't ignored
git status                     # review the list: NO data/, NO .venv/
git commit -m "Initial project skeleton: notebook, report, docs"
git log --oneline              # your first snapshot
```
Publish to GitHub as a **private** repo (academic integrity: never public):
```bash
gh repo create ARTI5001-A3-TTS --private --source=. --remote=origin --push
git remote -v                  # 'origin' now points to GitHub
```
Invite your teammates on the web page (repo → Settings → Collaborators → Add people), or from the terminal:
```bash
gh api -X PUT repos/<your-username>/ARTI5001-A3-TTS/collaborators/<teammate-username>
```

## Lesson 2: Teammates join
```bash
git clone https://github.com/<your-username>/ARTI5001-A3-TTS.git
cd ARTI5001-A3-TTS
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put LJSpeech in data/LJSpeech-1.1/ (it's ignored, so everyone downloads it themselves)
```

## Lesson 3: The daily team workflow (branches + pull requests)
Never work directly on `main`. One branch per task:
```bash
git switch main
git pull                                  # get everyone's latest work FIRST
git switch -c m1/text-normalisation       # create + move to a new branch
# ... edit files ...
git status
git diff                                  # see exactly what you changed (unstaged)
git add notebook/A3_TTS.ipynb
git diff --staged                         # see what will be committed
git commit -m "1.1: justify text normalisation, add trimming stats"
git push -u origin m1/text-normalisation  # -u links the branch to GitHub (first push only)
gh pr create --fill                       # open a pull request
gh pr list                                # see open PRs
gh pr checkout 3                          # test a teammate's PR locally
gh pr review 3 --approve
gh pr merge 3 --squash --delete-branch    # merge into main
git switch main && git pull               # update your main afterwards
git branch -d m1/text-normalisation       # delete the local branch
```
Branch naming: `m1/...`, `m2/...`, `m3/...` so ownership (and contribution evidence) is visible.

**Good commit messages:** start with the rubric item, e.g. `1.3: add expB config (3k utts)`, `report 2.2: system diagram`.

## Lesson 4: Inspecting history
```bash
git log --oneline --graph --all     # the whole branch tree
git log -p -- report/main.tex       # every change to one file
git show <commit>                   # one commit in full
git blame report/main.tex           # who wrote each line (contribution evidence!)
git shortlog -sn                    # commit count per author
git diff main..m2/pipeline          # compare two branches
```

## Lesson 5: Undoing things (from safest to most dangerous)
| Situation | Command |
|---|---|
| Discard unsaved edits to a file | `git restore file` |
| Unstage a file (keep the edits) | `git restore --staged file` |
| Fix the last commit message / add a forgotten file (before pushing) | `git commit --amend` |
| Undo a pushed commit safely (makes a new "reverse" commit) | `git revert <commit>` |
| Move the branch back but keep the changes | `git reset --soft HEAD~1` |
| Throw away commits **and** changes (local only, dangerous) | `git reset --hard <commit>` |
| "I lost something" | `git reflog` (git's undo history; almost nothing is truly lost) |

Rule: **never** `reset --hard` or `push --force` on `main`.

## Lesson 6: Switching tasks halfway through
```bash
git stash push -m "half-done metrics"   # shelve uncommitted work
git switch main                         # do something else
git switch m3/metrics
git stash list
git stash pop                           # bring it back
```

## Lesson 7: Merge conflicts (they will happen in main.tex)
```bash
git switch m2/system-design
git pull origin main          # or: git merge main
# CONFLICT in report/main.tex
git status                    # shows "both modified"
```
Open the file and look for:
```
<<<<<<< HEAD
your version
=======
their version
>>>>>>> main
```
Keep the right text, delete the markers, then:
```bash
git add report/main.tex
git commit                    # completes the merge
```
**Avoiding conflicts:** each person edits only their own sections; pull often; keep PRs small.
**Notebooks:** `.ipynb` is JSON, so its conflicts are painful. Before committing run
`jupyter nbconvert --clear-output --inplace notebook/A3_TTS.ipynb`, or agree that only one person
edits a given section at a time.

## Lesson 8: Rebase (cleaner history; once you're comfortable)
```bash
git switch m3/metrics
git fetch origin
git rebase origin/main        # replay your commits on top of the latest main
# on conflict: fix the file, git add <file>, git rebase --continue   (or git rebase --abort)
git push --force-with-lease   # needed after a rebase; ONLY on your own branch
```

## Lesson 9: Tag the submission
```bash
git tag -a v1.0-submitted -m "Submitted to Canvas 2 Nov 2026"
git push origin v1.0-submitted
```

## Lesson 10: Overleaf ↔ GitHub
Overleaf menu → GitHub → link this repo (needs an Overleaf plan with GitHub sync), **or** upload
`report/` as a zip and copy the final `.tex` back into the repo with a commit. Never edit the same
section in both places at once.

## Cheat sheet
```
status · add · commit -m · push · pull · switch -c · switch · branch -d · log --oneline --graph
diff · diff --staged · restore · restore --staged · revert · stash / stash pop · reflog · tag
```
