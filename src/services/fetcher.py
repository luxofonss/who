from pathlib import Path
from typing import Tuple

from git import Repo, GitCommandError, InvalidGitRepositoryError
from loguru import logger


REPOS_DIR = Path("storage/repos")


def clone_repo(project_id: str, github_url: str, branch: str = "main") -> Tuple[Path, str]:
    """Clone *or* pull a repository.

    Returns a tuple of (local_path, commit_sha).
    """
    repo_path = REPOS_DIR / project_id
    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    if repo_path.exists():
        logger.info(f"Repository {project_id} already exists – pulling latest changes …")
        try:
            repo = Repo(repo_path)
            # Fetch then checkout & pull.
            repo.git.fetch()
            repo.git.checkout(branch)
            repo.git.pull()
        except (GitCommandError, InvalidGitRepositoryError) as exc:
            logger.error(f"Failed updating existing repo: {exc}.  Re-cloning afresh.")
            # nuke and re-clone for demo robustness
            import shutil

            shutil.rmtree(repo_path)
            repo = Repo.clone_from(github_url, repo_path, branch=branch, depth=1)
    else:
        logger.info(f"Cloning {github_url} (branch={branch}) into {repo_path}")
        repo = Repo.clone_from(github_url, repo_path, branch=branch, depth=1)

    sha = repo.head.commit.hexsha
    logger.info(f"Repo ready at {repo_path} (HEAD={sha[:7]})")
    return repo_path, sha 