<!-- Pytest Coverage Comment:Begin -->
<a href="https://github.com/constructorfleet/mcp-plex/blob/main/README.md"><img alt="Coverage" src="https://img.shields.io/badge/Coverage-85%25-green.svg" /></a><details><summary>Coverage Report </summary><table><tr><th>File</th><th>Stmts</th><th>Miss</th><th>Cover</th><th>Missing</th></tr><tbody><tr><td colspan="5"><b>mcp_plex</b></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/constructorfleet/mcp-plex/blob/main/mcp_plex/loader.py">loader.py</a></td><td>172</td><td>45</td><td>74%</td><td><a href="https://github.com/constructorfleet/mcp-plex/blob/main/mcp_plex/loader.py#L139-L190">139&ndash;190</a>, <a href="https://github.com/constructorfleet/mcp-plex/blob/main/mcp_plex/loader.py#L295-L302">295&ndash;302</a>, <a href="https://github.com/constructorfleet/mcp-plex/blob/main/mcp_plex/loader.py#L340-L346">340&ndash;346</a>, <a href="https://github.com/constructorfleet/mcp-plex/blob/main/mcp_plex/loader.py#L450-L463">450&ndash;463</a></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/constructorfleet/mcp-plex/blob/main/mcp_plex/server.py">server.py</a></td><td>51</td><td>4</td><td>92%</td><td><a href="https://github.com/constructorfleet/mcp-plex/blob/main/mcp_plex/server.py#L32-L33">32&ndash;33</a>, <a href="https://github.com/constructorfleet/mcp-plex/blob/main/mcp_plex/server.py#L41">41</a>, <a href="https://github.com/constructorfleet/mcp-plex/blob/main/mcp_plex/server.py#L94">94</a></td></tr><tr><td><b>TOTAL</b></td><td><b>325</b></td><td><b>49</b></td><td><b>85%</b></td><td>&nbsp;</td></tr></tbody></table></details>
<!-- Pytest Coverage Comment:End -->

# mcp-plex

Plex MCP Tool Server.

## Loader

The `load-data` CLI can load Plex metadata into Qdrant. Run it once with:

```bash
uv run load-data --sample-dir sample-data
```

To keep the loader running continuously, pass `--continuous` and optionally set a
delay between runs:

```bash
uv run load-data --continuous --delay 600
```

## Docker

A Dockerfile is provided to build a GPU-enabled image based on
`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` using [uv](https://github.com/astral-sh/uv)
for dependency management. Build and run the loader:

```bash
docker build -t mcp-plex .
docker run --rm --gpus all mcp-plex --sample-dir /data
```

Use the `--continuous` and `--delay` flags with `docker run` to keep the loader
running in a loop.
