# 数学里最温柔的一个词 · 抖音竖屏短片

1080×1920，30fps，72 秒。全部由代码生成：numpy + Pillow 渲染 2.5D 撞色色块世界，numpy 合成钢琴配乐，ffmpeg 合成。

| 文件 | 作用 |
| --- | --- |
| `core.py` | 画布与渲染核心：正交相机、等轴测挤出色块（2× 超采样）、柔和投影、景深前景、光带、色差、纸张颗粒；文字系统（Noto CJK SC → DejaVu 回退、【】高亮）沿用 `intro_video/core.py` |
| `scenes.py` | 时间轴、镜头轨道、七幕：碎裂 / 狄利克雷点阵 / 黎曼柱 / 横切与数钱 / 板层与塔 / 测度为零 / 几乎处处 |
| `subs.py` | 双声部字幕：数学句（米白、钴蓝竖线）+ 生活句（加粗、芥末黄关键词），逐字弹入，避开底部 25% 与右侧 15% |
| `music.py` | numpy 合成钢琴：前半段单音留白，31–50 s 加和弦与低音，结尾长和弦渐弱 |
| `render.py` | 关键帧联系图、分块渲染（可断点续跑，带时间预算）、合成两个成片 |

```bash
python3 music.py                                   # build_audio/music.wav
LB_SCALE=0.5 python3 render.py keys s2 4 7.5 11    # 540×960 关键帧联系图（带安全区参考线）
LB_SCALE=0.5 python3 render.py all 540             # 540×960 低清预览，最多跑 540 s，重跑即续
LB_SCALE=0.5 python3 render.py mux
python3 render.py all 540 && python3 render.py mux # 1080×1920 正式版 → 最温柔的词.mp4 / 最温柔的词_无音乐.mp4
```

`LB_NFRAG` 控制碎片数量（默认 600，渲染太慢时调低，镜头不变），`NPROC` 控制进程数。
依赖：Python 3、numpy、Pillow、ffmpeg、fonts-noto-cjk、DejaVu。
