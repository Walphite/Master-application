#!/usr/bin/env python3
"""按学校名称 + 数学研究方向关键词，用 Claude 的网络搜索工具查找匹配的教授。"""

import argparse
import csv
import json
import sys

import anthropic

MODEL = "claude-opus-5"

SCHEMA = {
    "type": "object",
    "properties": {
        "professors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "research_area": {"type": "string"},
                    "homepage_url": {"type": "string"},
                    "source_url": {"type": "string"},
                },
                "required": ["name", "title", "research_area", "homepage_url", "source_url"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["professors"],
    "additionalProperties": False,
}

PROMPT_TEMPLATE = """请使用网络搜索工具，查找“{school}”数学系（或应用数学、统计等相近院系）中，
研究方向与关键词“{keyword}”匹配或高度相关的在职教授（含助理/副/正教授、讲席教授等）。

要求：
1. 只收录你能在搜索结果中找到可靠依据的教授，不要编造姓名或链接。
2. 优先访问该校数学系官网的教师名录（faculty/people 页面），核实姓名、研究方向、个人主页。
3. homepage_url 填教授本人的个人主页；如果没有独立主页，退而使用系里的教师介绍页，并在 title 字段注明"无独立主页"。
4. research_area 用简洁短语概括该教授与关键词相关的具体研究方向，不要照抄整段简介。
5. source_url 填你据以判断的信息来源页面（例如系教师名录页）。
6. 最多返回 15 位最相关的教授，按相关度从高到低排列；如果确实找不到匹配的教授，返回空列表。
"""


def search_professors(school: str, keyword: str, max_uses: int = 10, max_restarts: int = 3) -> list[dict]:
    client = anthropic.Anthropic()
    user_prompt = PROMPT_TEMPLATE.format(school=school, keyword=keyword)
    messages = [{"role": "user", "content": user_prompt}]

    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses}]
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}

    for _ in range(max_restarts + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            tools=tools,
            output_config=output_config,
            messages=messages,
        )

        if response.stop_reason == "refusal":
            detail = response.stop_details.explanation if response.stop_details else ""
            raise RuntimeError(f"请求被拒绝: {detail}")

        if response.stop_reason == "pause_turn":
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response.content},
            ]
            continue

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError("模型没有返回文本结果。")
        return json.loads(text).get("professors", [])

    raise RuntimeError("搜索被多次打断（pause_turn），仍未获得最终结果，可尝试调低 --max-uses。")


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("未找到匹配的教授。")
        return

    headers = ["姓名", "职称", "研究方向", "个人主页"]
    keys = ["name", "title", "research_area", "homepage_url"]
    widths = [max(len(h), *(len(r.get(k, "")) for r in rows)) for h, k in zip(headers, keys)]

    def fmt_row(cells):
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row([r.get(k, "") for k in keys]))


def save_csv(rows: list[dict], path: str) -> None:
    fieldnames = ["name", "title", "research_area", "homepage_url", "source_url"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="按学校 + 数学研究方向关键词搜索匹配的教授")
    parser.add_argument("school", help="学校名称，例如：斯坦福大学 / Stanford University")
    parser.add_argument("keyword", help="数学研究方向关键词，例如：代数几何 / algebraic geometry")
    parser.add_argument("--max-uses", type=int, default=10, help="web_search 工具最大调用次数（默认 10）")
    parser.add_argument("--csv", metavar="PATH", help="将结果另存为 CSV 文件")
    args = parser.parse_args()

    try:
        rows = search_professors(args.school, args.keyword, args.max_uses)
    except anthropic.APIStatusError as e:
        print(f"API 错误: {e.message}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("网络连接错误，请检查网络。", file=sys.stderr)
        sys.exit(1)

    print_table(rows)

    if args.csv:
        save_csv(rows, args.csv)
        print(f"\n已保存到 {args.csv}")


if __name__ == "__main__":
    main()
