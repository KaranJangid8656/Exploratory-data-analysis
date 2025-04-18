# utils/ai_summary.py
from openai import OpenAI
import pandas as pd

def get_ai_summary_openrouter(df: pd.DataFrame, api_key: str, referer="", title="EDA Dashboard") -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="your_api_key",
    )

    preview_data = df.head(10).to_string(index=False)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""You're a data analyst. Analyze this dataset preview:
{preview_data}

Summarize its structure, potential insights, issues (missing data, skewed columns), and how to further explore it."""
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages,
        extra_headers={
            "HTTP-Referer": referer,
            "X-Title": title
        },
        extra_body={}
    )

    return response.choices[0].message.content
