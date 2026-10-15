# 阿白 · 个人介绍视频

约 92 秒，1920×1080，30fps。全部由代码生成：numpy 浮点画布做加法混合与双层辉光，Pillow 排字，ffmpeg 合成。

| 文件 | 作用 |
| --- | --- |
| `core.py` | 画布、双线性 splat 加法混合、辉光、星空与星云、文字（CJK → DejaVu 回退）、HUD、字幕 |
| `scenes.py` | 七幕：初值 / 积分 / 分支树 / 进动 / 方向场 / 现在 / 结尾，以及字幕时间轴 |
| `music.py` | numpy 合成钢琴琶音（D–A–Bm–G 循环）+ 卷积混响 |
| `render.py` | 关键帧联系图、分块并行渲染（可断点续跑）、最终合成 |

```bash
python3 music.py                       # build/music.wav
python3 render.py keys s1 2 6 10       # 关键帧联系图 build/keys_s1.png
python3 render.py all                  # 分块渲染 build/chunks/*.mp4（已完成的块会跳过）
python3 render.py mux                  # 合成 阿白.mp4
```

依赖：Python 3、numpy、Pillow、ffmpeg、Noto Serif/Sans CJK SC、DejaVu（含 Serif Italic）。
