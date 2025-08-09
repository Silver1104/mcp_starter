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
    description="Analyze CSV/Excel files and create charts. Just provide the file data and optionally describe what you want to see.",
    use_when="User uploads a data file (CSV/Excel) and wants analysis or charts.",
    side_effects="Returns data summary and automatically generated charts as images."
)

@mcp.tool(description=ANALYZE_DATA_DESCRIPTION.model_dump_json())
async def analyze_data_file(
    file_data: Annotated[str, Field(description="Base64-encoded file data from user upload")],
    analysis_request: Annotated[str, Field(description="What the user wants to analyze or see (optional)")] = "general overview",
    file_type: Annotated[str, Field(description="File format: 'csv' or 'xlsx'")] = "csv"
) -> List[TextContent | ImageContent]:
    """
    Simplified data analysis tool that's easier for AI assistants to use.
    Automatically detects what charts to make based on the data structure.
    """
    try:
        # Decode the file data
        try:
            file_bytes = base64.b64decode(file_data)
        except Exception:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid base64 file data"))

        buf = io.BytesIO(file_bytes)

        # Load dataframe with better error handling
        df = None
        if file_type.lower() == "csv":
            # Try multiple approaches for CSV loading
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            separators = [',', ';', '\t']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        buf.seek(0)
                        df = pd.read_csv(buf, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:  # Success if we got multiple columns
                            break
                    except:
                        continue
                if df is not None and len(df.columns) > 1:
                    break
                    
        elif file_type.lower() == "xlsx":
            try:
                buf.seek(0)
                df = pd.read_excel(buf)
            except Exception as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to read Excel file: {str(e)}"))
        
        if df is None or df.empty:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Could not load data from file or file is empty"))

        # Clean up column names (remove extra spaces, etc.)
        df.columns = df.columns.astype(str).str.strip()
        
        # Basic info about the dataset
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Build summary
        summary_parts = [
            f"📊 Data Analysis Results",
            f"",
            f"Dataset Overview:",
            f"• Rows: {n_rows:,}",
            f"• Columns: {n_cols}",
            f"• Numeric columns: {len(numeric_cols)}",
            f"• Text columns: {len(text_cols)}",
            f"",
            f"Column Names: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}",
            f""
        ]
        
        # Add basic statistics for numeric columns
        if numeric_cols:
            summary_parts.append("Key Statistics:")
            for col in numeric_cols[:5]:  # Show stats for first 5 numeric columns
                try:
                    mean_val = df[col].mean()
                    summary_parts.append(f"• {col}: avg={mean_val:.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")
                except:
                    pass
        
        results = [TextContent(type="text", text="\n".join(summary_parts))]
        
        # AUTO-GENERATE USEFUL CHARTS
        charts_created = 0
        max_charts = 4

        def _render_figure_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        
        # Chart 1: If we have numeric data, create a correlation heatmap
        if len(numeric_cols) >= 2 and charts_created < max_charts:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f', ax=ax)
                ax.set_title('Correlation Between Numeric Variables')
                plt.tight_layout()
                
                chart_b64 = _render_figure_to_base64(fig)
                results.append(ImageContent(type="image", mimeType="image/png", data=chart_b64))
                charts_created += 1
            except Exception as e:
                print(f"Failed to create correlation heatmap: {e}")
        
        # Chart 2: Distribution of first numeric column
        if numeric_cols and charts_created < max_charts:
            try:
                col = numeric_cols[0]
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create histogram with better styling
                df[col].hist(bins=30, alpha=0.7, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                chart_b64 = _render_figure_to_base64(fig)
                results.append(ImageContent(type="image", mimeType="image/png", data=chart_b64))
                charts_created += 1
            except Exception as e:
                print(f"Failed to create histogram: {e}")
        
        # Chart 3: If we have categorical + numeric, create a bar chart
        if text_cols and numeric_cols and charts_created < max_charts:
            try:
                cat_col = text_cols[0]
                num_col = numeric_cols[0]
                
                # Group by categorical and average the numeric
                grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                
                # Limit to top 15 categories for readability
                if len(grouped) > 15:
                    grouped = grouped.head(15)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                grouped.plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title(f'Average {num_col} by {cat_col}')
                ax.set_xlabel(cat_col)
                ax.set_ylabel(f'Average {num_col}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                chart_b64 = _render_figure_to_base64(fig)
                results.append(ImageContent(type="image", mimeType="image/png", data=chart_b64))
                charts_created += 1
            except Exception as e:
                print(f"Failed to create bar chart: {e}")
        
        # Chart 4: If we have 2+ numeric columns, create a scatter plot
        if len(numeric_cols) >= 2 and charts_created < max_charts:
            try:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[x_col], df[y_col], alpha=0.6, color='green')
                ax.set_title(f'{y_col} vs {x_col}')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                chart_b64 = _render_figure_to_base64(fig)
                results.append(ImageContent(type="image", mimeType="image/png", data=chart_b64))
                charts_created += 1
            except Exception as e:
                print(f"Failed to create scatter plot: {e}")
        
        # Add summary of what was created
        final_summary = f"\n✅ Generated {charts_created} charts automatically based on your data structure."
        if charts_created == 0:
            final_summary = "\n⚠️ No charts could be generated - this might be a text-only dataset."
        
        results[0] = TextContent(type="text", text=results[0].text + final_summary)
        
        return results

    except McpError:
        raise
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        return [TextContent(type="text", text=error_msg)]
    
@mcp.tool
async def debug_data_upload(
    file_data: Annotated[str, Field(description="Base64 file data to inspect")]
) -> str:
    """Simple tool to check if file upload is working correctly"""
    try:
        decoded = base64.b64decode(file_data)
        size_kb = len(decoded) / 1024
        
        # Try to detect file type
        if decoded.startswith(b'PK'):
            file_type = "Excel/ZIP file"
        elif b',' in decoded[:1000] or b';' in decoded[:1000]:
            file_type = "Likely CSV"
        else:
            file_type = "Unknown format"
            
        return f"✅ File received successfully!\nSize: {size_kb:.1f} KB\nDetected type: {file_type}\nFirst 100 bytes: {decoded[:100]}"
    except Exception as e:
        return f"❌ File upload failed: {str(e)}"

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
