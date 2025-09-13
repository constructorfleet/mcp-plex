# AGENTS

## Architecture
- `mcp_plex/loader.py` ingests Plex, TMDb, and IMDb metadata, relies on Qdrant to generate dense and sparse embeddings, and stores items in a Qdrant collection.
- `mcp_plex/server.py` exposes retrieval and search tools via FastMCP backed by Qdrant.
- `mcp_plex/types.py` defines the Pydantic models used across the project.
- When making architectural design decisions, add a short note here describing the decision and its rationale.
- Embedding generation was moved from local FastEmbed models to Qdrant's document API to reduce local dependencies and centralize vector creation.
- Actor names are stored as a top-level payload field and indexed in Qdrant to enable actor and year-based filtering.
- Dense and sparse embedding model names are configurable via `DENSE_MODEL` and
  `SPARSE_MODEL` environment variables or the corresponding CLI options.
- Hybrid search uses Qdrant's built-in `FusionQuery` with reciprocal rank fusion
  to combine dense and sparse results before optional cross-encoder reranking.

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
- Use [uv](https://github.com/astral-sh/uv) for all Python dependency management and command execution.
- Install project and development dependencies with:
  ```bash
  uv sync --extra dev
  ```

## Versioning
- Always bump the version in `pyproject.toml` for any change.
- Update `uv.lock` after version or dependency changes by running `uv lock`.

## Checks
- Run linting with `uv run ruff check .`.
- Run the test suite with `uv run pytest` and ensure it passes before committing.

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
