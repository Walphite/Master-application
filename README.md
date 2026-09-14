# 数学教授搜索工具

按学校名称 + 数学研究方向关键词，调用 Anthropic Claude 的网络搜索工具，
在该校数学系（或应用数学/统计等相近院系）中查找研究方向匹配的教授，
输出姓名、职称、研究方向、个人主页链接的表格。

## 安装

```bash
pip install -r requirements.txt
```

## 配置

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 使用

```bash
python find_professors.py "斯坦福大学" "代数几何"

# 调整搜索轮次上限，并导出 CSV
python find_professors.py "Stanford University" "algebraic geometry" --max-uses 15 --csv result.csv
```

## 说明

- 结果基于模型的网络搜索，可能不完整或有误，建议核对个人主页链接后再使用。
- `--max-uses` 控制单次运行中 `web_search` 工具的最大调用次数，学校教师较多、方向较冷门时可以调高。
