# AGENTS

## Architecture
- The project is organized into dedicated `loader`, `server`, and `common` packages under `mcp_plex/`.
- Package-specific architectural notes live alongside the code in `mcp_plex/loader/AGENTS.md`, `mcp_plex/server/AGENTS.md`, and `mcp_plex/common/AGENTS.md`.
- Update this file when repo-wide conventions or folder-level guidelines change.
- Review the Architecture Decision Records in `docs/adr/` before implementing changes that affect system design, and document
  new architectural decisions with an ADR.

## User Queries
The project should handle natural-language searches and recommendations such as:
- "What's that show that had the billionaire that invited the artist and the journalist and the musician to his house and ended up with an alien that breaks free from a meteor?"
- "Find a horror movie similar to Schindler's List"
- "Recommend an action comedy movie with Tom Holland"
- "What new movies do I have?"
- "Any new shows?"
- "Find the newest movie with Tom Cruise"
- "Suggest a movie from the 90's with Glenn Close"

## Dependency Management
- Use [uv](https://github.com/astral-sh/uv) for all Python dependency management and command execution; do not fall back to `pip`, `poetry`, or other tools.
- When inspecting installed packages, look inside the active uv-managed virtual environment (e.g., `.venv/`) rather than the system Python directories.
- Install project and development dependencies with:
  ```bash
  uv sync --extra dev
  ```

## Versioning
- Always bump the version in `pyproject.toml` for any change.
- Mirror the version change in `docker/pyproject.deps.toml` so the Docker dependency manifest stays consistent with the root project file.
- Update `uv.lock` after version or dependency changes by running `uv lock`.

## Checks
- Run linting with `uv run ruff check .`.
- Run the test suite with `uv run pytest`, ensure it passes, and address any warnings emitted by the run before committing.

## Testing Practices
- Use realistic (or as realistic as possible) data in tests; avoid meaningless placeholder values.
- Always test both positive and negative logical paths.
- Do **not** use `# pragma: no cover`; add tests to exercise code paths instead.
- All changes should include tests that demonstrate the new or modified behavior.

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
