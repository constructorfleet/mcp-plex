# AGENTS

## Dependency Management
- Use [uv](https://github.com/astral-sh/uv) for all Python dependency management and command execution.
- Install project and development dependencies with:
  ```bash
  uv sync --extra dev
  ```

## Checks
- Run linting with `uv run ruff check .`.
- Run the test suite with `uv run pytest` and ensure it passes before committing.

## Efficiency and Search
- Use `rg` (ripgrep) for recursive search.
- Avoid `ls -R` and `grep -R` as they generate excessive output.

## Git Commit Guidelines
- Commit messages must follow [Conventional Commit](https://www.conventionalcommits.org/) standards.
- Keep commits atomic and focused.

## Pull Request Guidelines
Each PR description should clearly state:
1. **What** the PR does.
2. **Why** the change is needed.
3. **Affects**: what parts of the project are impacted.
4. **Testing**: commands and results proving the change works.
5. **Documentation**: note any docs updated or confirm none needed.
