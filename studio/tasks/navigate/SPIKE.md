# M3-1 技術驗證：空地估計用什麼方法（2026-11-22）

## 問題

TASKS M3-1：比較「單眼深度模型」（候選 Depth Anything V2 Small）和「起步時正前方是地板」的顏色比對法，推論 ≥ 8 FPS 才採用深度模型。板子是 **Jetson Nano 4 GB**（使用者 2026-11-22 確認）：CPU 4 核 A57、GPU 128 核 Maxwell、RAM 和 GPU 共用。

## 兩個方法

| | 地板顏色比對（`freespace.py`） | 單眼深度模型（Depth Anything V2 Small） |
|---|---|---|
| 需要訓練／權重 | 不用 | 25M 參數的 ViT-S，權重約 100 MB |
| 執行位置 | CPU（OpenCV） | GPU（TensorRT FP16） |
| RAM | 幾 MB | 引擎＋啟動 TensorRT 約 300~600 MB（估計），和車道線模型、torch 搶 4 GB |
| 會出錯的情況 | 地板和障礙物顏色很像、地板顏色不均（地毯花紋、強烈陰影） | 反光地板、低紋理牆面；輸出是相對深度，還是要再判斷「地板」 |
| JetPack 4 的風險 | 無 | ViT 的 LayerNorm／attention 要走 ONNX → trtexec（TensorRT 8.0），舊版 torch2trt 不支援；輸入 252 以下才有機會接近即時 |

## 量測

`python3 -m studio.tasks.navigate.spike`（中位數，每格 200 次）：

| 機器 | 輸入 | 地板比對 | 追蹤器 |
|---|---|---|---|
| sandbox PC（x86，12 核） | 224 | 0.33 ms | 1.23 ms |
| | 256 | 0.39 ms | 1.95 ms |
| | 320 | 0.89 ms | 2.00 ms |
| docker 限 4 核、Python 3.6、OpenCV 4.5 | 224 | 0.33 ms | 1.20 ms |
| | 320 | 0.53 ms | 2.87 ms |
| **Jetson Nano** | 224 / 256 / 320 | ⏳ 待量 | ⏳ 待量 |
| **Jetson Nano，深度模型（trtexec FP16）** | 252 / 336 | ⏳ 待量（需要先匯出 ONNX） | — |

A57 大約比這台 PC 慢 5~10 倍，估計地板比對＋追蹤在 Nano 上每幀 < 30 ms，遠低於 8 FPS 的 125 ms 預算。

深度模型沒有量：sandbox 連 huggingface.co 需要網路核准（`Approval required for huggingface.co`），權重下載不了。要量的話：在 PC 上匯出 ONNX（輸入 252×252、opset 11~13），複製到車上，執行 `python3 -m studio.tasks.navigate.spike --depth-onnx <檔案>`。

## 決定

- **預設用地板顏色比對**（不需要模型，CPU 幾毫秒，不佔 GPU／RAM，車道線模型可以照常跑）。
- 「正前方太近就停車」的規則建立在它上面，**不依賴任何模型**（TASKS M3-4 的最後一道保護）。
- 學習版（M3-6）用 project4 的網路學「使用者會點哪裡」，取代規則版的規劃，但停車規則保留。
- 深度模型保留成之後的選項：等 Nano 上量到 ≥ 8 FPS（而且 RAM 夠）再接進 `freespace.py` 的同一個介面（每個扇區給 0~1 的可通行距離）。

## 驗收（TASKS M3-2：20 張人工標註的圖、「前方有障礙」正確率 ≥ 90%）

- 合成圖（地板紋理＋白膠帶＋箱子，10 張前方有障礙、10 張沒有／在旁邊）：**20/20**（`studio/tests/test_navigate.py`）。
- ⏳ 真實照片：還沒有。拍法：車子放在場地上，前方放／不放箱子各拍 10 張（箱子距離 10~40 cm），存成資料夾後可以用 `freespace.py` + `planner.py` 批次判斷。
