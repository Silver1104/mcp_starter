
import asyncio
import os
import base64
import io
import re
from typing import Annotated, List, Tuple
from dotenv import load_dotenv

# MCP / FastMCP - Updated imports to fix deprecation warning
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier  # Updated from deprecated bearer auth
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
# Simple JWT auth provider (updated from deprecated bearer auth)
# ------------------------------------------------------------------
# For now, let's skip auth to avoid complexity - you can add it back later
auth_provider = None  # Simplified - add JWT auth later if needed

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
# Helper function to convert matplotlib figures to base64
# ------------------------------------------------------------------
def _render_figure_to_base64(fig: matplotlib.figure.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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
# WhatsApp-compatible data analysis tool
# ------------------------------------------------------------------
def detect_file_type(file_bytes: bytes) -> str:
    # XLSX (Office Open XML) files are ZIP archives starting with 'PK'
    if file_bytes[:2] == b'PK':
        return "xlsx"
    # Simple check for binary Excel (.xls) - starts with D0 CF 11 E0
    if file_bytes[:4] == b'\xD0\xCF\x11\xE0':
        return "xls"
    return "csv"  # default fallback

@mcp.tool(description="Analyze CSV/Excel files uploaded via WhatsApp. Automatically detects file uploads and creates charts.")
async def analyze_data_file(
    file_data: Annotated[str, Field(description="Base64-encoded file data")] = None,
    puch_file_data: Annotated[str, Field(description="File data from Puch upload")] = None,
    data: Annotated[str, Field(description="Raw data content")] = None,
    content: Annotated[str, Field(description="File content")] = None,
    message: Annotated[str, Field(description="User's analysis request")] = "analyze this data",
    user_message: Annotated[str, Field(description="What user said about the data")] = "analyze this data",
    query: Annotated[str, Field(description="User query")] = "analyze this data",
    file_type: Annotated[str, Field(description="csv or xlsx")] = "auto",
    filename: Annotated[str, Field(description="Original filename if available")] = None,
) -> List[TextContent | ImageContent]:
    try:
        # Detect file data from parameters
        actual_file_data = None
        for param in [file_data, puch_file_data, data, content]:
            if param and len(param) > 100:
                actual_file_data = param
                break
        
        if not actual_file_data:
            return [TextContent(type="text", text="❌ No file data detected. Please upload a CSV/XLSX file.")]

        # Decode Base64
        try:
            file_bytes = base64.b64decode(actual_file_data)
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Could not decode file data: {str(e)}")]

        buf = io.BytesIO(file_bytes)

        # Auto-detect file type if not specified
        if file_type == "auto":
            if filename:
                if filename.lower().endswith('.xlsx'):
                    file_type = "xlsx"
                elif filename.lower().endswith('.csv'):
                    file_type = "csv"
                else:
                    file_type = detect_file_type(file_bytes)
            else:
                file_type = detect_file_type(file_bytes)

        # Load DataFrame
        df = None
        loading_method = ""

        if file_type == "csv":
            # Try multiple CSV reading attempts
            for method_name, kwargs in [
                ("UTF-8, comma", {'encoding': 'utf-8', 'sep': ','}),
                ("UTF-8, semicolon", {'encoding': 'utf-8', 'sep': ';'}),
                ("UTF-8, tab", {'encoding': 'utf-8', 'sep': '\t'}),
                ("Latin1, comma", {'encoding': 'latin1', 'sep': ','}),
                ("Auto-detect", {'sep': None, 'engine': 'python'}),
            ]:
                try:
                    buf.seek(0)
                    df = pd.read_csv(buf, **kwargs)
                    if len(df.columns) > 1 and not df.empty:
                        loading_method = method_name
                        break
                except:
                    pass

            # If CSV failed, try Excel fallback (misnamed file)
            if df is None:
                try:
                    buf.seek(0)
                    df = pd.read_excel(buf)
                    loading_method = "Excel fallback (misnamed CSV)"
                except:
                    pass

        elif file_type in ["xlsx", "xls"]:
            try:
                buf.seek(0)
                df = pd.read_excel(buf)
                loading_method = "Excel"
            except:
                # Fallback to CSV read if Excel fails
                try:
                    buf.seek(0)
                    df = pd.read_csv(buf)
                    loading_method = "CSV fallback (misnamed Excel)"
                except:
                    pass

        # If still failed
        if df is None or df.empty:
            return [TextContent(type="text", text="❌ Could not load file as CSV or Excel.")]

        # Clean columns
        df.columns = df.columns.astype(str).str.strip()

        # Prepare summary
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()

        summary = [
            f"📊 Data Analysis Complete",
            f"✅ Loaded with method: {loading_method}",
            f"📈 Shape: {n_rows:,} rows × {n_cols} columns",
            f"🔢 Numeric columns: {len(numeric_cols)} ({', '.join(numeric_cols[:5])})",
            f"📝 Text columns: {len(text_cols)} ({', '.join(text_cols[:5])})",
            "",
            f"User request: '{message or user_message or query}'",
        ]

        # Add preview
        try:
            preview = df.head(3).to_string(max_cols=6, max_colwidth=20)
            summary.append("\n📋 First few rows:\n" + preview)
        except:
            summary.append("\n(Preview unavailable)")

        results = [TextContent(type="text", text="\n".join(summary))]

        # Charts
        charts_made = 0
        max_charts = 3

        # Line chart
        if numeric_cols and charts_made < max_charts:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                date_col = next((col for col in df.columns if any(w in col.lower() for w in ['date', 'time', 'year', 'month'])), None)
                if date_col and date_col not in numeric_cols:
                    for num_col in numeric_cols[:3]:
                        ax.plot(df[date_col], df[num_col], marker='o', label=num_col)
                    ax.set_xlabel(date_col)
                else:
                    for num_col in numeric_cols[:3]:
                        ax.plot(df.index, df[num_col], marker='o', label=num_col)
                    ax.set_xlabel("Index")
                ax.legend()
                ax.set_title("Line Chart")
                plt.xticks(rotation=45)
                results.append(ImageContent(type="image", mimeType="image/png", data=_render_figure_to_base64(fig)))
                charts_made += 1
            except Exception as e:
                print(f"Line chart failed: {e}")

        # Bar chart
        if text_cols and numeric_cols and charts_made < max_charts:
            try:
                grouped = df.groupby(text_cols[0])[numeric_cols[0]].mean().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(12, 6))
                grouped.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f"{numeric_cols[0]} by {text_cols[0]}")
                plt.xticks(rotation=45)
                results.append(ImageContent(type="image", mimeType="image/png", data=_render_figure_to_base64(fig)))
                charts_made += 1
            except Exception as e:
                print(f"Bar chart failed: {e}")

        # Histogram
        if numeric_cols and charts_made < max_charts:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[numeric_cols[0]].hist(bins=20, ax=ax, color='lightgreen', edgecolor='black')
                ax.set_title(f"Distribution of {numeric_cols[0]}")
                results.append(ImageContent(type="image", mimeType="image/png", data=_render_figure_to_base64(fig)))
                charts_made += 1
            except Exception as e:
                print(f"Histogram failed: {e}")

        return results

    except Exception as e:
        return [TextContent(type="text", text=f"❌ Analysis failed: {str(e)}")]
    
# ------------------------------------------------------------------
# Simple Plain Text Sentiment Analysis Tool
# ------------------------------------------------------------------
SENTIMENT_ANALYSIS_DESCRIPTION = RichToolDescription(
    description="Analyze sentiment of reviews or text entered as plain text. Supports single reviews or multiple reviews separated by newlines.",
    use_when="Use when user types or pastes reviews/text for sentiment analysis.",
    side_effects="Returns sentiment scores and visual analysis."
)

@mcp.tool(description=SENTIMENT_ANALYSIS_DESCRIPTION.model_dump_json())
async def analyze_text_sentiment(
    text: Annotated[str, Field(description="Plain text reviews or content to analyze")] = None,
    content: Annotated[str, Field(description="Alternative text content parameter")] = None,
    message: Annotated[str, Field(description="User message containing text to analyze")] = None,
    reviews: Annotated[str, Field(description="Reviews text to analyze")] = None,
) -> List[TextContent | ImageContent]:
    """
    Analyze sentiment of plain text reviews. Automatically detects single vs multiple reviews.
    """
    
    # Get the actual text to analyze from any parameter
    actual_text = text or content or message or reviews
    if not actual_text or len(actual_text.strip()) < 5:
        return [TextContent(type="text", text="❌ Please provide text or reviews to analyze.")]
    
    try:
        # Clean and prepare the text
        actual_text = actual_text.strip()
        
        # Check if it's multiple reviews (contains newlines or common separators)
        potential_reviews = []
        
        # Try different separators
        for separator in ['\n', '||', '---', '###']:
            if separator in actual_text:
                potential_reviews = [r.strip() for r in actual_text.split(separator) if r.strip()]
                break
        
        # If no separators found, treat as single text but check for sentence breaks
        if not potential_reviews:
            # Look for review-like patterns (sentences ending with periods followed by capitals)
            import re
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', actual_text)
            if len(sentences) > 1 and any(len(s.split()) > 10 for s in sentences):
                potential_reviews = [s.strip() for s in sentences if len(s.strip()) > 10]
            else:
                potential_reviews = [actual_text]
        
        # Analyze each review/text segment
        results_data = []
        total_polarity = 0
        total_subjectivity = 0
        
        for i, review_text in enumerate(potential_reviews[:50], 1):  # Limit to 50 reviews
            if len(review_text) < 5:
                continue
                
            blob = TextBlob(review_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment_label = "Positive 😊"
                emoji = "😊"
            elif polarity < -0.1:
                sentiment_label = "Negative 😔"
                emoji = "😔"
            else:
                sentiment_label = "Neutral 😐"
                emoji = "😐"
            
            results_data.append({
                'index': i,
                'text': review_text[:100] + ('...' if len(review_text) > 100 else ''),
                'full_text': review_text,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment_label,
                'emoji': emoji
            })
            
            total_polarity += polarity
            total_subjectivity += subjectivity
        
        if not results_data:
            return [TextContent(type="text", text="❌ No valid text found to analyze.")]
        
        # Calculate overall statistics
        avg_polarity = total_polarity / len(results_data)
        avg_subjectivity = total_subjectivity / len(results_data)
        
        positive_count = sum(1 for r in results_data if r['polarity'] > 0.1)
        negative_count = sum(1 for r in results_data if r['polarity'] < -0.1)
        neutral_count = len(results_data) - positive_count - negative_count
        
        # Determine overall sentiment
        if avg_polarity > 0.1:
            overall_sentiment = "Positive 😊"
        elif avg_polarity < -0.1:
            overall_sentiment = "Negative 😔"
        else:
            overall_sentiment = "Neutral 😐"
        
        # Build results text
        if len(results_data) == 1:
            # Single review analysis
            result = results_data[0]
            analysis_text = [
                "🎭 **Sentiment Analysis Results**",
                "",
                f"📝 **Text:** {result['full_text'][:200]}{'...' if len(result['full_text']) > 200 else ''}",
                "",
                f"📊 **Sentiment:** {result['sentiment']}",
                f"📈 **Polarity Score:** {result['polarity']:.3f} (-1=Negative, +1=Positive)",
                f"📝 **Subjectivity:** {result['subjectivity']:.3f} (0=Objective, 1=Subjective)",
                f"🎯 **Confidence:** {min(abs(result['polarity']) * 100, 100):.1f}%",
            ]
        else:
            # Multiple reviews analysis
            analysis_text = [
                f"🎭 **Sentiment Analysis - {len(results_data)} Reviews**",
                "",
                f"📊 **Overall Sentiment:** {overall_sentiment}",
                f"📈 **Average Polarity:** {avg_polarity:.3f}",
                f"📝 **Average Subjectivity:** {avg_subjectivity:.3f}",
                "",
                f"😊 **Positive Reviews:** {positive_count} ({positive_count/len(results_data)*100:.1f}%)",
                f"😐 **Neutral Reviews:** {neutral_count} ({neutral_count/len(results_data)*100:.1f}%)",
                f"😔 **Negative Reviews:** {negative_count} ({negative_count/len(results_data)*100:.1f}%)",
                "",
                "📋 **Individual Reviews:**"
            ]
            
            # Add individual review results (first 10)
            for result in results_data[:10]:
                analysis_text.append(f"{result['emoji']} **{result['index']}.** [{result['polarity']:.2f}] {result['text']}")
            
            if len(results_data) > 10:
                analysis_text.append(f"... and {len(results_data) - 10} more reviews")
        
        results = [TextContent(type="text", text="\n".join(analysis_text))]
        
        # Create visualization
        try:
            if len(results_data) == 1:
                # Single review visualization
                fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle('Sentiment Analysis', fontsize=14, fontweight='bold')
                
                result = results_data[0]
                
                # Sentiment gauge
                colors = ['red' if result['polarity'] < 0 else 'yellow' if result['polarity'] == 0 else 'green']
                ax1.pie([abs(result['polarity']), 1-abs(result['polarity'])], 
                       colors=[colors[0], 'lightgray'],
                       startangle=90)
                ax1.set_title(f'{result["sentiment"]}\nScore: {result["polarity"]:.3f}')
                
                # Metrics bar chart
                metrics = ['Polarity', 'Subjectivity']
                values = [result['polarity'], result['subjectivity']]
                colors_bar = ['blue', 'orange']
                
                bars = ax2.bar(metrics, values, color=colors_bar, alpha=0.7)
                ax2.set_ylim(-1, 1)
                ax2.set_title('Sentiment Metrics')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height >= 0 else height - 0.05,
                           f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
                
            else:
                # Multiple reviews visualization
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'Sentiment Analysis - {len(results_data)} Reviews', fontsize=14)
                
                # 1. Sentiment distribution
                sizes = [positive_count, neutral_count, negative_count]
                labels = ['Positive', 'Neutral', 'Negative']
                colors = ['green', 'gray', 'red']
                
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Sentiment Distribution')
                
                # 2. Polarity histogram
                polarities = [r['polarity'] for r in results_data]
                ax2.hist(polarities, bins=min(20, len(results_data)), color='blue', alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Polarity Score')
                ax2.set_ylabel('Number of Reviews')
                ax2.set_title('Polarity Distribution')
                ax2.axvline(x=avg_polarity, color='red', linestyle='--', label=f'Average: {avg_polarity:.3f}')
                ax2.legend()
                
                # 3. Review index vs polarity
                indices = [r['index'] for r in results_data]
                ax3.plot(indices, polarities, 'o-', color='blue', alpha=0.7)
                ax3.set_xlabel('Review Number')
                ax3.set_ylabel('Polarity Score')
                ax3.set_title('Sentiment Trend')
                ax3.grid(True, alpha=0.3)
                ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                
                # 4. Summary statistics
                stats_text = f"""
Reviews Analyzed: {len(results_data)}

Average Scores:
• Polarity: {avg_polarity:.3f}
• Subjectivity: {avg_subjectivity:.3f}

Distribution:
• Positive: {positive_count} ({positive_count/len(results_data)*100:.1f}%)
• Neutral: {neutral_count} ({neutral_count/len(results_data)*100:.1f}%)
• Negative: {negative_count} ({negative_count/len(results_data)*100:.1f}%)

Overall: {overall_sentiment}
                """.strip()
                
                ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.axis('off')
                ax4.set_title('Summary Statistics')
            
            plt.tight_layout()
            results.append(ImageContent(type="image", mimeType="image/png", 
                                      data=_render_figure_to_base64(fig)))
            
        except Exception as chart_error:
            results.append(TextContent(type="text", 
                                     text=f"⚠️ Chart generation failed: {str(chart_error)}"))
        
        return results
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Sentiment analysis failed: {str(e)}")]

# ------------------------------------------------------------------
# Simple debugging tool (fixed - no **kwargs)
# ------------------------------------------------------------------
@mcp.tool(description="Debug tool to see what parameters Puch sends when uploading files")
async def debug_whatsapp_upload(
    file_data: Annotated[str, Field(description="Possible file data parameter")] = None,
    puch_file_data: Annotated[str, Field(description="Possible Puch file data")] = None,
    data: Annotated[str, Field(description="Possible data parameter")] = None,
    content: Annotated[str, Field(description="Possible content parameter")] = None,
    message: Annotated[str, Field(description="User message")] = None,
    filename: Annotated[str, Field(description="Filename if provided")] = None,
) -> str:
    """Debug tool to see exactly what Puch sends when you upload a file"""
    debug_lines = ["🔍 Debug: WhatsApp Upload Parameters", ""]
    
    params = {
        "file_data": file_data,
        "puch_file_data": puch_file_data,
        "data": data,
        "content": content,
        "message": message,
        "filename": filename
    }
    
    for key, value in params.items():
        if value is None:
            debug_lines.append(f"❌ {key}: None")
        elif isinstance(value, str):
            if len(value) > 100:
                debug_lines.append(f"✅ {key}: {len(value)} chars (looks like file data)")
                # Show first few characters to help identify format
                debug_lines.append(f"   Preview: {value[:50]}...")
            else:
                debug_lines.append(f"📝 {key}: '{value}'")
        else:
            debug_lines.append(f"📋 {key}: {type(value)} = {str(value)[:100]}")
    
    return "\n".join(debug_lines)

# ------------------------------------------------------------------
# Run server (use PORT env variable)
# ------------------------------------------------------------------
async def main():
    print(f"🚀 Starting MCP server on 0.0.0.0:{PORT}")
    # FastMCP internal transport name kept as "streamable-http" (works for demo)
    await mcp.run_async("streamable-http", host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    asyncio.run(main())
