param (
    [string]$Message = "Update project"
)

git add .
$status = git status --porcelain
if ($status) {
    git commit -m $Message
    git push origin main
    Write-Host "Successfully committed and pushed changes to GitHub." -ForegroundColor Green
} else {
    Write-Host "No changes to commit. Everything is up to date." -ForegroundColor Yellow
}
