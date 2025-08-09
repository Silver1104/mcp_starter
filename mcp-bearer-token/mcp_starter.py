# server.py
import asyncio
import os
import base64
import io
import re
from typing import Annotated, List, Tuple
from dotenv import load_dotenv

# MCP / FastMCP
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR

# Data & plotting
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# HTTP client + html helpers
import httpx
import markdownify
import readabilipy
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, AnyUrl

# load .env if present
load_dotenv()

# ------------------------------------------------------------------
# Environment / config
# ------------------------------------------------------------------
TOKEN = os.environ.get("AUTH_TOKEN")  # optional — if missing, we won't enforce auth on init
MY_NUMBER = os.environ.get("MY_NUMBER", "")
PORT = int(os.environ.get("PORT", "8080"))

# Print warnings, but don't crash (so initialize works)
if not TOKEN:
    print("⚠️  AUTH_TOKEN not set — running without enforced bearer auth. Set AUTH_TOKEN in env for production.")
if not MY_NUMBER:
    print("⚠️  MY_NUMBER not set — validate() will return an error string until configured.")

# ------------------------------------------------------------------
# Simple Bearer auth provider (optional)
# ------------------------------------------------------------------
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token and token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# Use auth provider only if TOKEN present
auth_provider = SimpleBearerAuthProvider(TOKEN) if TOKEN else None

# ------------------------------------------------------------------
# MCP server instance
# ------------------------------------------------------------------
mcp = FastMCP("Job Finder MCP Server", auth=auth_provider)

# ------------------------------------------------------------------
# Utility classes & functions (Fetch + HTML -> Markdown)
# ------------------------------------------------------------------
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(cls, url: str, user_agent: str = USER_AGENT, force_raw: bool = False) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(url, follow_redirects=True, headers={"User-Agent": user_agent})
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))
            if resp.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status {resp.status_code}"))
            page_raw = resp.text
            content_type = resp.headers.get("content-type", "")
            is_html = "text/html" in content_type
            if is_html and not force_raw:
                return cls.extract_content_from_html(page_raw), ""
            return page_raw, f"Content type {content_type} could not be simplified, returning raw content.\n"

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        try:
            ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
            if not ret or not ret.get("content"):
                return "<error>Failed to simplify HTML</error>"
            return markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        except Exception:
            # fallback: strip tags
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text("\n")
            return text[:20000]  # limit length

    @staticmethod
    async def duckduckgo_search_links(query: str, num_results: int = 5) -> list[str]:
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Search failed</error>"]
            soup = BeautifulSoup(resp.text, "html.parser")
            links = []
            for a in soup.find_all("a", class_="result__a", href=True):
                href = a["href"]
                if href.startswith("http"):
                    links.append(href)
                if len(links) >= num_results:
                    break
            return links or ["<error>No results found</error>"]

# ------------------------------------------------------------------
# Tool descriptions
# ------------------------------------------------------------------
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# ------------------------------------------------------------------
# Required: validate tool (Puch needs this)
# ------------------------------------------------------------------
@mcp.tool
async def validate() -> str:
    if not MY_NUMBER:
        # return an error string so Puch sees something but not empty
        return "<error>MY_NUMBER not configured on server</error>"
    return MY_NUMBER

# ------------------------------------------------------------------
# Image tool: convert to B/W (unchanged)
# ------------------------------------------------------------------
MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert base64 image to black and white PNG.",
    use_when="Use when user uploads an image to convert to BW.",
    side_effects="Returns a PNG image (base64)."
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(puch_image_data: Annotated[str, Field(description="Base64-encoded image data")] = None) -> List[ImageContent]:
    from PIL import Image
    if not puch_image_data:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="No image data provided"))
    try:
        b = base64.b64decode(puch_image_data)
        img = Image.open(io.BytesIO(b))
        bw = img.convert("L")
        buf = io.BytesIO()
        bw.save(buf, format="PNG")
        return [ImageContent(type="image", mimeType="image/png", data=base64.b64encode(buf.getvalue()).decode("utf-8"))]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# ------------------------------------------------------------------
# Job finder tool (kept, minor robustness)
# ------------------------------------------------------------------
JobFinderDescription = RichToolDescription(
    description="Analyze job descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links."
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="User's goal or search query")],
    job_description: Annotated[str | None, Field(description="Optional job description text")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="Optional job posting URL")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML if True")] = False,
) -> str:
    if job_description:
        return f"📝 Job analysis for goal: {user_goal}\n\n---\n{job_description.strip()}\n---"
    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return f"🔗 Fetched: {job_url}\n\n---\n{content.strip()}\n---"
    if "find" in user_goal.lower() or "look for" in user_goal.lower():
        links = await Fetch.duckduckgo_search_links(user_goal)
        return "🔍 Results:\n" + "\n".join(f"- {l}" for l in links)
    raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide job_description, job_url or a search query."))

# ------------------------------------------------------------------
# ANALYZE DATA FILE: accepts base64 file_data OR file_url pointing to csv/xlsx
# ------------------------------------------------------------------
ANALYZE_DATA_DESCRIPTION = RichToolDescription(
    description="Accept CSV/XLSX (base64 or URL), run analytics, and return summary + PNG images (base64).",
    use_when="User uploads a spreadsheet and asks for charts or analysis.",
    side_effects="Reads file, processes it, and returns text + images."
)

def _detect_intents_and_targets(user_request: str, df: pd.DataFrame) -> dict:
    req = (user_request or "").lower()
    chart_types = []
    for k, v in (("bar", "bar"), ("line", "line"), ("scatter", "scatter"), ("hist", "hist"), ("heatmap", "heatmap")):
        if k in req:
            chart_types.append(v)
    if not chart_types:
        chart_types = ["line"]
    agg = None
    for w in ("sum", "total", "average", "avg", "mean", "count"):
        if w in req:
            agg = "mean" if w in ("average", "avg", "mean") else ("count" if w == "count" else "sum")
            break
    top_n = None
    m = re.search(r"top\s+(\d+)", req)
    if m:
        top_n = int(m.group(1))
    mentioned = []
    for col in df.columns:
        if re.search(rf"\b{re.escape(col.lower())}\b", req):
            mentioned.append(col)
    group_by = None
    metric = None
    m2 = re.search(r"by\s+([a-z0-9_ ]+)", req)
    if m2:
        candidate = m2.group(1).strip()
        for col in df.columns:
            if candidate in col.lower():
                group_by = col
                break
    if mentioned:
        if len(mentioned) >= 2:
            group_by, metric = mentioned[0], mentioned[1]
        elif len(mentioned) == 1:
            c = mentioned[0]
            if pd.api.types.is_numeric_dtype(df[c]):
                metric = c
                for cc in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[cc]):
                        group_by = cc
                        break
            else:
                group_by = c
                for cc in df.columns:
                    if pd.api.types.is_numeric_dtype(df[cc]):
                        metric = cc
                        break
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if metric is None and numeric_cols:
        metric = numeric_cols[0]
    if group_by is None and object_cols:
        group_by = object_cols[0] if object_cols else None
    return {"chart_types": chart_types, "agg": agg, "top_n": top_n, "group_by": group_by, "metric": metric}

def _render_figure_to_base64(fig: matplotlib.figure.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Chart helper functions (use seaborn where helpful)
def _make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return _render_figure_to_base64(fig)

def _make_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return _render_figure_to_base64(fig)

def _make_scatter(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return _render_figure_to_base64(fig)

def _make_hist(df: pd.DataFrame, columns: List[str], title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    df[columns].hist(ax=ax)
    fig.suptitle(title)
    plt.tight_layout()
    return _render_figure_to_base64(fig)

def _make_heatmap_correlation(df: pd.DataFrame, title: str) -> str:
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return _render_figure_to_base64(fig)

@mcp.tool(description=ANALYZE_DATA_DESCRIPTION.model_dump_json())
async def analyze_data_file(
    file_data: Annotated[str, Field(description="Base64-encoded CSV or XLSX file data")] = None,
    file_type: Annotated[str, Field(description="csv or xlsx")] = "csv",
    file_url: Annotated[str | None, Field(description="Optional: public URL to download the file")] = None,
    user_request: Annotated[str, Field(description="Natural language analysis request")] = "",
    sheet_name: Annotated[str | None, Field(description="Optional sheet name for xlsx")] = None,
    max_charts: Annotated[int, Field(description="Max charts to return")] = 3,
) -> List[TextContent | ImageContent]:
    try:
        # obtain bytes: either from base64 payload or by fetching URL
        if file_url:
            # allow remote fetch (S3, presigned URL); supports csv/xlsx
            raw_text, _ = await Fetch.fetch_url(file_url)
            # If it's a CSV text, convert to bytes
            file_bytes = raw_text.encode("utf-8")
            # try to infer file_type if not supplied
            if not file_type:
                if file_url.lower().endswith(".xlsx") or "sheet" in file_url.lower():
                    file_type = "xlsx"
                elif file_url.lower().endswith(".csv"):
                    file_type = "csv"
        elif file_data:
            try:
                file_bytes = base64.b64decode(file_data)
            except Exception:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="file_data is not valid base64"))
        else:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide file_data (base64) or file_url"))

        buf = io.BytesIO(file_bytes)

        # load dataframe
        if file_type.lower() == "csv":
            # try with utf-8, fallback to latin1; infer sep if needed
            try:
                df = pd.read_csv(buf)
            except Exception:
                buf.seek(0)
                try:
                    df = pd.read_csv(buf, encoding="latin1")
                except Exception:
                    buf.seek(0)
                    # try sniffing delimiter
                    sample = buf.read(4096).decode(errors="ignore")
                    sep = "," if sample.count(",") >= sample.count("\t") else "\t"
                    buf.seek(0)
                    df = pd.read_csv(buf, sep=sep, engine="python")
        elif file_type.lower() == "xlsx":
            try:
                import openpyxl  # ensure driver exists
                buf.seek(0)
                df = pd.read_excel(buf, sheet_name=sheet_name) if sheet_name else pd.read_excel(buf)
                if isinstance(df, dict):
                    # multiple sheets returned -> pick first
                    df = list(df.values())[0]
            except Exception as e:
                raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Failed to read xlsx: {e}"))
        else:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Unsupported file type; use 'csv' or 'xlsx'"))

        if df is None or df.empty:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Uploaded file contained no data"))

        # derive basic metadata
        n_rows, n_cols = df.shape
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        object_cols = df.select_dtypes(include="object").columns.tolist()

        parsed = _detect_intents_and_targets(user_request or "", df)
        chart_types = parsed["chart_types"]
        agg = parsed["agg"]
        top_n = parsed["top_n"]
        group_by = parsed["group_by"]
        metric = parsed["metric"]

        # build summary text
        summary_lines = [
            f"Rows: {n_rows}, Columns: {n_cols}",
            f"Columns: {', '.join(columns)}",
            f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}",
            f"Detected chart types: {', '.join(chart_types)}",
            f"Detected aggregation: {agg or 'none'}",
            f"Group by: {group_by or 'none'}, Metric: {metric or 'none'}",
        ]

        chart_images: List[Tuple[str, str]] = []
        charts_generated = 0

        # Aggregation path
        if agg and group_by and metric:
            try:
                if agg == "count":
                    agg_df = df.groupby(group_by).size().reset_index(name="count").sort_values("count", ascending=False)
                    metric_col = "count"
                else:
                    agg_df = getattr(df.groupby(group_by)[metric], agg)().reset_index()
                    metric_col = metric
                    agg_df = agg_df.sort_values(metric_col, ascending=False)
                if top_n:
                    agg_df = agg_df.head(top_n)
                title = f"{agg.title()} of {metric_col} by {group_by}"
                chart_images.append((title, _make_bar_chart(agg_df, group_by, metric_col, title)))
                charts_generated += 1
                summary_lines.append(f"Aggregated rows: {len(agg_df)} (top_n={top_n})")
            except Exception:
                # if aggregation fails, continue to other charts
                pass

        # other chart generation
        for ctype in chart_types:
            if charts_generated >= max_charts:
                break
            if ctype == "heatmap":
                if len(numeric_cols) >= 2:
                    chart_images.append(("Correlation heatmap", _make_heatmap_correlation(df[numeric_cols], "Correlation heatmap")))
                    charts_generated += 1
            elif ctype == "hist":
                cols = (numeric_cols[:4] or columns[:4])
                chart_images.append((f"Histogram of {', '.join(cols)}", _make_hist(df, cols, f"Histogram of {', '.join(cols)}")))
                charts_generated += 1
            elif ctype == "scatter":
                if metric and pd.api.types.is_numeric_dtype(df[metric]):
                    x_col = next((c for c in numeric_cols if c != metric), None)
                    if x_col:
                        chart_images.append((f"Scatter: {x_col} vs {metric}", _make_scatter(df, x_col, metric, f"Scatter: {x_col} vs {metric}")))
                        charts_generated += 1
                elif len(numeric_cols) >= 2:
                    chart_images.append((f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}", _make_scatter(df, numeric_cols[0], numeric_cols[1], f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")))
                    charts_generated += 1
            elif ctype == "line":
                if numeric_cols:
                    y = numeric_cols[0]
                    x = (object_cols[0] if object_cols else None)
                    if x:
                        try:
                            chart_images.append((f"Line: {y} by {x}", _make_line_chart(df, x, y, f"Line: {y} by {x}")))
                            charts_generated += 1
                        except Exception:
                            fig, ax = plt.subplots(figsize=(8, 5)); df[y].plot(ax=ax); ax.set_title(f"Line: {y}"); plt.tight_layout()
                            chart_images.append((f"Line: {y}", _render_figure_to_base64(fig))); charts_generated += 1
                    else:
                        fig, ax = plt.subplots(figsize=(8, 5)); df[y].plot(ax=ax); ax.set_title(f"Line: {y}"); plt.tight_layout()
                        chart_images.append((f"Line: {y}", _render_figure_to_base64(fig))); charts_generated += 1
            elif ctype == "bar":
                if group_by and metric:
                    bar_df = df.groupby(group_by)[metric].sum().reset_index().sort_values(metric, ascending=False)
                    if top_n:
                        bar_df = bar_df.head(top_n)
                    chart_images.append((f"Sum of {metric} by {group_by}", _make_bar_chart(bar_df, group_by, metric, f"Sum of {metric} by {group_by}")))
                    charts_generated += 1
                elif numeric_cols:
                    fig, ax = plt.subplots(figsize=(8, 5)); df[numeric_cols[0]].plot(kind="bar", ax=ax); ax.set_title(f"Bar: {numeric_cols[0]}"); plt.tight_layout()
                    chart_images.append((f"Bar: {numeric_cols[0]}", _render_figure_to_base64(fig))); charts_generated += 1

        # fallback charts
        if charts_generated == 0:
            if len(numeric_cols) >= 2:
                chart_images.append((f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}", _make_scatter(df, numeric_cols[0], numeric_cols[1], f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")))
            elif numeric_cols:
                fig, ax = plt.subplots(figsize=(8, 5)); df[numeric_cols[0]].plot(ax=ax); ax.set_title(f"Line: {numeric_cols[0]}"); plt.tight_layout()
                chart_images.append((f"Line: {numeric_cols[0]}", _render_figure_to_base64(fig)))
            else:
                summary_lines.append("No numeric columns available to plot.")

        # statistical summary (short)
        try:
            stats = df.describe(include="all").head(12).to_string()
            summary_lines.append("\nStatistical summary (first lines):")
            summary_lines.extend(stats.splitlines()[:12])
        except Exception:
            pass

        # build results
        results: List[TextContent | ImageContent] = []
        results.append(TextContent(type="text", text="\n".join(summary_lines)))
        for title, b64 in chart_images[:max_charts]:
            results.append(ImageContent(type="image", mimeType="image/png", data=b64))

        return results

    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# ------------------------------------------------------------------
# Health route for quick checks (optional: used by Render or manual)
# Note: fastmcp provides its own transport; this health endpoint is useful for external checks.
# ------------------------------------------------------------------
# @mcp.endpoint(path="/health")
# async def health() -> dict:
#     return {"status": "ok", "name": "Job Finder MCP Server", "version": "1.0.0"}

# ------------------------------------------------------------------
# Run server (use PORT env variable)
# ------------------------------------------------------------------
async def main():
    print(f"🚀 Starting MCP server on 0.0.0.0:{PORT}")
    # FastMCP internal transport name kept as "streamable-http" (works for demo)
    await mcp.run_async("streamable-http", host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    asyncio.run(main())
