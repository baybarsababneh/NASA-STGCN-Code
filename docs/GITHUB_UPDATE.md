# Updating GitHub

Keep the existing repository private. This ZIP contains no project Git history, and the private remote was not accessible during review. Its older commits, branches, tags, releases, Actions logs, issues, and attachments have not been cleared for publication. A clean latest commit does not remove old content.

## Check the download

On macOS, put the ZIP and its checksum file together:

```bash
shasum -a 256 -c NASA_STGCN_Public_Reviewed_SHA256.txt
unzip NASA_STGCN_Public_Reviewed.zip
cd NASA-STGCN-Code
python3 tools/verify_archive.py
```

Use Python 3.11 or later. The ZIP extracts one `NASA-STGCN-Code/` directory. Read `docs/PUBLIC_RELEASE_REVIEW.md` and choose reuse terms for original code before calling the release open source. Do not upload the companion data/checkpoint archives until their rights are established.

## Update the existing private repository

Replace `OWNER/PRIVATE_REPO` and the absolute source path below. Use a fresh clone so unrelated local work is protected. These commands update files on a review branch; they do not change visibility.

```bash
git clone https://github.com/OWNER/PRIVATE_REPO.git nasa-private-review
cd nasa-private-review
git fetch --all --tags
git bundle create ../nasa-private-before-review.bundle --all
git switch -c public-release-review
rsync -an --delete --exclude='.git/' /absolute/path/NASA-STGCN-Code/ ./
```

Read the dry-run output. It should remove old files absent from the reviewed ZIP, including old large payloads, while preserving `.git/`. Keep the bundle private. Then apply the same copy:

```bash
rsync -a --delete --exclude='.git/' /absolute/path/NASA-STGCN-Code/ ./
python3 tools/verify_archive.py
git status --short
git diff --stat
git diff --check
git add -A
git diff --cached --stat
git commit -m "Prepare reviewed EXP001–EXP015 research archive"
git push -u origin public-release-review
```

Review the branch on GitHub before merging. This preserves history, including any historical material that should remain private. It does not make the existing repository safe to expose.

## Publish without importing private history

The simplest route is a separate empty repository, initially **private**, containing only the reviewed tree. Keep the old repository private. Create the new repository on GitHub without an initial README, licence, or gitignore. In a fresh extraction of the ZIP:

```bash
cd /absolute/path/NASA-STGCN-Code
python3 tools/verify_archive.py
git init -b main
git add -A
git diff --cached --stat
git commit -m "Archive EXP001–EXP015 source and recorded results"
git remote add origin https://github.com/OWNER/NEW_REPO.git
git push -u origin main
```

Replace `OWNER/NEW_REPO`. Check the new private repository's files and author email, resolve any ownership/licence decision, and then use Settings → General → Danger Zone → Change repository visibility. Do not copy old branches, tags, releases, issues, or Actions artefacts into it.

If retaining the old history publicly is essential, audit the complete remote first. `git fetch --all --tags`, `git log --all --format=fuller`, and `git rev-list --objects --all` expose the reachable commit and object inventory. Scan every reachable blob for credentials and private references, and review GitHub-hosted metadata separately. Rotate any exposed credentials. History removal requires a separately reviewed rewrite and coordination with collaborators; an ordinary deletion commit is insufficient. No force-push or history rewrite was performed for this review.
