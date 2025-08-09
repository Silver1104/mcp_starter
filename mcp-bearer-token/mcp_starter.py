# import asyncio
# from typing import Annotated
# import os
# from dotenv import load_dotenv
# from fastmcp import FastMCP
# from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
# from mcp import ErrorData, McpError
# from mcp.server.auth.provider import AccessToken
# from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
# from pydantic import BaseModel, Field, AnyUrl

# import markdownify
# import httpx
# import readabilipy

# # --- Load environment variables ---
# load_dotenv()

# TOKEN = os.environ.get("AUTH_TOKEN")
# MY_NUMBER = os.environ.get("MY_NUMBER")

# assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
# assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# # --- Auth Provider ---
# class SimpleBearerAuthProvider(BearerAuthProvider):
#     def __init__(self, token: str):
#         k = RSAKeyPair.generate()
#         super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
#         self.token = token

#     async def load_access_token(self, token: str) -> AccessToken | None:
#         if token == self.token:
#             return AccessToken(
#                 token=token,
#                 client_id="puch-client",
#                 scopes=["*"],
#                 expires_at=None,
#             )
#         return None

# # --- Rich Tool Description model ---
# class RichToolDescription(BaseModel):
#     description: str
#     use_when: str
#     side_effects: str | None = None

# # --- Fetch Utility Class ---
# class Fetch:
#     USER_AGENT = "Puch/1.0 (Autonomous)"

#     @classmethod
#     async def fetch_url(
#         cls,
#         url: str,
#         user_agent: str,
#         force_raw: bool = False,
#     ) -> tuple[str, str]:
#         async with httpx.AsyncClient() as client:
#             try:
#                 response = await client.get(
#                     url,
#                     follow_redirects=True,
#                     headers={"User-Agent": user_agent},
#                     timeout=30,
#                 )
#             except httpx.HTTPError as e:
#                 raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

#             if response.status_code >= 400:
#                 raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

#             page_raw = response.text

#         content_type = response.headers.get("content-type", "")
#         is_page_html = "text/html" in content_type

#         if is_page_html and not force_raw:
#             return cls.extract_content_from_html(page_raw), ""

#         return (
#             page_raw,
#             f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
#         )

#     @staticmethod
#     def extract_content_from_html(html: str) -> str:
#         """Extract and convert HTML content to Markdown format."""
#         ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
#         if not ret or not ret.get("content"):
#             return "<error>Page failed to be simplified from HTML</error>"
#         content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
#         return content

#     @staticmethod
#     async def google_search_links(query: str, num_results: int = 5) -> list[str]:
#         """
#         Perform a scoped DuckDuckGo search and return a list of job posting URLs.
#         (Using DuckDuckGo because Google blocks most programmatic scraping.)
#         """
#         ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
#         links = []

#         async with httpx.AsyncClient() as client:
#             resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
#             if resp.status_code != 200:
#                 return ["<error>Failed to perform search.</error>"]

#         from bs4 import BeautifulSoup
#         soup = BeautifulSoup(resp.text, "html.parser")
#         for a in soup.find_all("a", class_="result__a", href=True):
#             href = a["href"]
#             if "http" in href:
#                 links.append(href)
#             if len(links) >= num_results:
#                 break

#         return links or ["<error>No results found.</error>"]

# # --- MCP Server Setup ---
# mcp = FastMCP(
#     "Job Finder MCP Server",
#     auth=SimpleBearerAuthProvider(TOKEN),
# )

# # --- Tool: validate (required by Puch) ---
# @mcp.tool
# async def validate() -> str:
#     return MY_NUMBER

# # --- Tool: job_finder (now smart!) ---
# JobFinderDescription = RichToolDescription(
#     description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
#     use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
#     side_effects="Returns insights, fetched job descriptions, or relevant job links.",
# )

# @mcp.tool(description=JobFinderDescription.model_dump_json())
# async def job_finder(
#     user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
#     job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
#     job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
#     raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
# ) -> str:
#     """
#     Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
#     """
#     if job_description:
#         return (
#             f"📝 **Job Description Analysis**\n\n"
#             f"---\n{job_description.strip()}\n---\n\n"
#             f"User Goal: **{user_goal}**\n\n"
#             f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
#         )

#     if job_url:
#         content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
#         return (
#             f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
#             f"---\n{content.strip()}\n---\n\n"
#             f"User Goal: **{user_goal}**"
#         )

#     if "look for" in user_goal.lower() or "find" in user_goal.lower():
#         links = await Fetch.google_search_links(user_goal)
#         return (
#             f"🔍 **Search Results for**: _{user_goal}_\n\n" +
#             "\n".join(f"- {link}" for link in links)
#         )

#     raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# # Image inputs and sending images

# MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
#     description="Convert an image to black and white and save it.",
#     use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
#     side_effects="The image will be processed and saved in a black and white format.",
# )

# @mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
# async def make_img_black_and_white(
#     puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
# ) -> list[TextContent | ImageContent]:
#     import base64
#     import io

#     from PIL import Image

#     try:
#         image_bytes = base64.b64decode(puch_image_data)
#         image = Image.open(io.BytesIO(image_bytes))

#         bw_image = image.convert("L")

#         buf = io.BytesIO()
#         bw_image.save(buf, format="PNG")
#         bw_bytes = buf.getvalue()
#         bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

#         return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# # --- Run MCP Server ---
# async def main():
#     print("🚀 Starting MCP server on http://0.0.0.0:8086")
#     await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

# if __name__ == "__main__":
#     asyncio.run(main())


# ---------- Full integrated MCP server with enhanced analyze_data_file tool ----------
import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# New imports for analysis
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
import numpy as np
from typing import List, Tuple

# ensure matplotlib backend that works in headless environments
matplotlib.use("Agg")

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (previously defined) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# Image inputs and sending images (existing tool)
MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# ---------- NEW: Enhanced analyze_data_file tool (NL -> Pandas + multiple charts) ----------

ANALYZE_DATA_DESCRIPTION = RichToolDescription(
    description="Accept a CSV/XLSX file, perform advanced analytics (group-by/aggregation/top-N/correlation/summary) and return charts + text summary.",
    use_when="Use when user uploads a spreadsheet and asks for analytics or charts in plain language.",
    side_effects="Reads file, processes it, and returns text summary and PNG images (base64)."
)


def _detect_intents_and_targets(user_request: str, df: pd.DataFrame) -> dict:
    """
    Very-lightweight NL parser:
    - detects chart types requested (bar/line/scatter/hist/heatmap)
    - detects aggregation intents (sum/mean/count/avg)
    - detects group-by column and metric column names mentioned
    - detects "top N" requests
    """
    req = user_request.lower()

    # chart types
    chart_types = []
    if "bar" in req:
        chart_types.append("bar")
    if "line" in req:
        chart_types.append("line")
    if "scatter" in req:
        chart_types.append("scatter")
    if "hist" in req or "distribution" in req:
        chart_types.append("hist")
    if "heatmap" in req or "correlation" in req:
        chart_types.append("heatmap")
    if not chart_types:
        chart_types = ["line"]  # default

    # aggregation
    agg = None
    if "sum" in req:
        agg = "sum"
    elif "total" in req:
        agg = "sum"
    elif "average" in req or "avg" in req or "mean" in req:
        agg = "mean"
    elif "count" in req:
        agg = "count"

    # top N
    top_n = None
    m = re.search(r"top\s+(\d+)", req)
    if m:
        top_n = int(m.group(1))

    # find possible columns mentioned
    mentioned = []
    for col in df.columns:
        if re.search(rf"\b{re.escape(col.lower())}\b", req):
            mentioned.append(col)

    # heuristics for group and metric
    group_by = None
    metric = None
    # if "by <col>" pattern
    m2 = re.search(r"by\s+([a-z0-9_ ]+)", req)
    if m2:
        candidate = m2.group(1).strip()
        for col in df.columns:
            if candidate in col.lower():
                group_by = col
                break

    # if not found, try to assign from mentioned columns
    if mentioned:
        if len(mentioned) >= 2:
            group_by, metric = mentioned[0], mentioned[1]
        elif len(mentioned) == 1:
            # choose 1st mentioned as group if it's non-numeric, else metric
            c = mentioned[0]
            if pd.api.types.is_numeric_dtype(df[c]):
                metric = c
                # choose a non-numeric as group_by if exists
                for cc in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[cc]):
                        group_by = cc
                        break
            else:
                group_by = c
                # choose a numeric column as metric
                for cc in df.columns:
                    if pd.api.types.is_numeric_dtype(df[cc]):
                        metric = cc
                        break

    # if still missing, fill with sensible defaults
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if metric is None and numeric_cols:
        metric = numeric_cols[0]
    if group_by is None and object_cols:
        group_by = object_cols[0]

    return {
        "chart_types": chart_types,
        "agg": agg,
        "top_n": top_n,
        "group_by": group_by,
        "metric": metric,
    }


def _render_figure_to_base64(fig: matplotlib.figure.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    return _render_figure_to_base64(fig)


def _make_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="line", x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    return _render_figure_to_base64(fig)


def _make_scatter(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="scatter", x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
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
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return _render_figure_to_base64(fig)


@mcp.tool(description=ANALYZE_DATA_DESCRIPTION.model_dump_json())
async def analyze_data_file(
    file_data: Annotated[str, Field(description="Base64-encoded CSV or XLSX file data")],
    file_type: Annotated[str, Field(description="Type of the file: 'csv' or 'xlsx'")] = "csv",
    user_request: Annotated[str, Field(description="Natural language description of analysis/graph the user wants")] = "",
    sheet_name: Annotated[str | None, Field(description="(Optional) Excel sheet name to read, if xlsx")] = None,
    max_charts: Annotated[int, Field(description="Max number of charts to return (default 3)")] = 3,
) -> list[TextContent | ImageContent]:
    """
    Enhanced analytics tool:
    - Decodes base64 file_data, reads into Pandas
    - Uses simple NL heuristics to determine aggregation/grouping/charting
    - Produces summary text + up to `max_charts` images (PNG base64)
    """
    try:
        # decode
        file_bytes = base64.b64decode(file_data)
        buf = io.BytesIO(file_bytes)

        # load df
        if file_type.lower() == "csv":
            df = pd.read_csv(buf)
        elif file_type.lower() == "xlsx":
            # allow sheet_name param if present
            df = pd.read_excel(buf, sheet_name=sheet_name) if sheet_name else pd.read_excel(buf)
            # if sheet_name returned a dict (multiple sheets), pick first
            if isinstance(df, dict):
                # pick first sheet's data
                first_key = list(df.keys())[0]
                df = df[first_key]
        else:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Unsupported file type. Use 'csv' or 'xlsx'."))

        if df is None or df.empty:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Uploaded file contained no data."))

        # basic info
        n_rows, n_cols = df.shape
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        object_cols = df.select_dtypes(include="object").columns.tolist()

        # parse request
        parsed = _detect_intents_and_targets(user_request or "", df)
        chart_types = parsed["chart_types"]
        agg = parsed["agg"]
        top_n = parsed["top_n"]
        group_by = parsed["group_by"]
        metric = parsed["metric"]

        results: List[TextContent | ImageContent] = []

        # create descriptive summary
        summary_lines = [
            f"Rows: {n_rows}, Columns: {n_cols}",
            f"Columns: {', '.join(columns)}",
            f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}",
            f"Detected chart types: {', '.join(chart_types)}",
            f"Detected aggregation: {agg or 'none'}",
            f"Group by: {group_by or 'none'}, Metric: {metric or 'none'}",
        ]

        # If aggregation specified and group_by + metric exist, produce aggregated table
        df_for_plot = df.copy()
        charts_generated = 0
        chart_images: List[Tuple[str, str]] = []  # (title, base64)

        if agg and group_by and metric:
            if agg == "count":
                agg_df = df.groupby(group_by).size().reset_index(name="count").sort_values("count", ascending=False)
                metric_col = "count"
            else:
                agg_df = getattr(df.groupby(group_by)[metric], agg)().reset_index()
                metric_col = metric
                agg_df = agg_df.sort_values(metric_col, ascending=False)
            # apply top_n if requested
            if top_n:
                agg_df = agg_df.head(top_n)
            # prepare plot
            title = f"{agg.title()} of {metric_col} by {group_by}"
            chart_images.append((title, _make_bar_chart(agg_df, group_by, metric_col, title)))
            charts_generated += 1
            summary_lines.append(f"Aggregated rows: {len(agg_df)} (top_n={top_n})")

        # if no aggregation or user asked other charts, generate charts from numeric columns
        # support multiple charts requested
        for ctype in chart_types:
            if charts_generated >= max_charts:
                break

            if ctype == "heatmap":
                if len(numeric_cols) >= 2:
                    title = "Correlation heatmap"
                    chart_images.append((title, _make_heatmap_correlation(df[numeric_cols], title)))
                    charts_generated += 1
            elif ctype == "hist":
                # choose up to 4 numeric cols
                cols = (numeric_cols[:4] or columns[:4])
                title = f"Histogram of {', '.join(cols)}"
                chart_images.append((title, _make_hist(df, cols, title)))
                charts_generated += 1
            elif ctype == "scatter":
                # scatter needs two numeric columns; prefer metric & another numeric
                if metric and pd.api.types.is_numeric_dtype(df[metric]):
                    # find second numeric for x
                    x_col = None
                    for c in numeric_cols:
                        if c != metric:
                            x_col = c
                            break
                    if x_col:
                        title = f"Scatter: {x_col} vs {metric}"
                        chart_images.append((title, _make_scatter(df, x_col, metric, title)))
                        charts_generated += 1
                elif len(numeric_cols) >= 2:
                    title = f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}"
                    chart_images.append((title, _make_scatter(df, numeric_cols[0], numeric_cols[1], title)))
                    charts_generated += 1
            elif ctype == "line":
                # use first numeric as y and index or first column as x
                if len(numeric_cols) >= 1:
                    y = numeric_cols[0]
                    # choose x column if object or date-like exists, otherwise use index
                    x = None
                    for c in object_cols:
                        x = c
                        break
                    if x:
                        title = f"Line: {y} by {x}"
                        try:
                            chart_images.append((title, _make_line_chart(df, x, y, title)))
                            charts_generated += 1
                        except Exception:
                            # fallback to plotting y alone (index vs y)
                            fig, ax = plt.subplots(figsize=(8, 5))
                            df[y].plot(ax=ax)
                            ax.set_title(title)
                            plt.tight_layout()
                            chart_images.append((title, _render_figure_to_base64(fig)))
                            charts_generated += 1
                    else:
                        # index vs y
                        fig, ax = plt.subplots(figsize=(8, 5))
                        df[y].plot(ax=ax)
                        ax.set_title(f"Line: {y}")
                        plt.tight_layout()
                        chart_images.append((f"Line: {y}", _render_figure_to_base64(fig)))
                        charts_generated += 1
            elif ctype == "bar":
                # if group_by exists, use aggregation suitable for bar
                if group_by and metric:
                    # reuse previous aggregated df if exists
                    if agg:
                        # already aggregated above; skip
                        pass
                    else:
                        bar_df = df.groupby(group_by)[metric].sum().reset_index().sort_values(metric, ascending=False)
                        if top_n:
                            bar_df = bar_df.head(top_n)
                        title = f"Sum of {metric} by {group_by}"
                        chart_images.append((title, _make_bar_chart(bar_df, group_by, metric, title)))
                        charts_generated += 1
                elif len(numeric_cols) >= 1:
                    # simple bar of first numeric column (index)
                    y = numeric_cols[0]
                    fig, ax = plt.subplots(figsize=(8, 5))
                    df[y].plot(kind="bar", ax=ax)
                    ax.set_title(f"Bar: {y}")
                    plt.tight_layout()
                    chart_images.append((f"Bar: {y}", _render_figure_to_base64(fig)))
                    charts_generated += 1

        # If no charts generated yet, fallback: plot top 2 numeric columns
        if charts_generated == 0:
            if len(numeric_cols) >= 2:
                title = f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}"
                chart_images.append((title, _make_scatter(df, numeric_cols[0], numeric_cols[1], title)))
            elif numeric_cols:
                title = f"Line: {numeric_cols[0]}"
                fig, ax = plt.subplots(figsize=(8, 5))
                df[numeric_cols[0]].plot(ax=ax)
                ax.set_title(title)
                plt.tight_layout()
                chart_images.append((title, _render_figure_to_base64(fig)))
            else:
                # cannot make a chart
                summary_lines.append("No numeric columns available to plot.")

        # Statistical summary
        try:
            stats = df.describe(include="all").to_string()
            summary_lines.append("\nStatistical summary (first lines):")
            summary_lines.extend(stats.splitlines()[:12])  # first 12 lines to keep short
        except Exception:
            pass

        # Build final return
        results.append(TextContent(type="text", text="\n".join(summary_lines)))

        # attach images (up to max_charts)
        for i, (title, b64) in enumerate(chart_images[:max_charts]):
            results.append(ImageContent(type="image", mimeType="image/png", data=b64))

        return results

    except McpError:
        # re-raise MCP errors directly
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
# ---------- End integrated server ----------
