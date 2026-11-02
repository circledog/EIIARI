# Project 4：AI 道路辨識

JetBot 道路跟隨：模型從相機畫面預測「要前往的目標點 (x, y)」，再用 PD + bias 控制左右輪。

```
PC / Colab   Data Loader → Preprocessing → Data Augmentation → Model training → Model compression (DepGraph)
                                │ 設定存進模型 .json                                     │ .pth / .onnx
Jetson Nano  Camera Input → Preprocessing（同一套）→ Model inferencing (PyTorch / TensorRT) → Motor controlling (PD + bias)
```

## 檔案一覽

| 檔案 | 在哪裡跑 | 用途 |
|---|---|---|
| `train_model.ipynb` | — | 課程原始範例，留作參考 |
| `data_loader.py` | PC | 資料集與 DataLoader |
| `preprocessing.py` | PC + Nano | 前處理：把車道線（白膠帶）標成紅色，彎道也適用；附預覽工具 |
| `augmentation.py` | PC（Nano 的「輸入干擾」也會用到） | 資料增強，三組開關 |
| `model.py` | PC + Nano | 模型定義、載入 |
| `compression.py` | PC（`wrap_trt_engine` 在 Nano） | DepGraph 剪枝、ONNX 匯出、TRT 包裝 |
| `camera.py` | Nano | 相機輸入與前處理 |
| `control.py` | Nano | PD + bias 差速控制 |
| `train.py` | PC | 訓練 → 剪枝 → finetune → 匯出 |
| `live_demo.ipynb` | Nano | 推論＋控制的操作介面 |
| `requirements-pc.txt` | PC | PC / Colab 端套件 |

資料放在 `data/`、模型放在 `models/`、行駛紀錄放在 `runs/`，這三個資料夾都不進 git。

## 各模組簡介

### `data_loader.py`

- `XYDataset(directory=None, augmenter=None)`：讀取 `xy_<x>_<y>_<uuid>.jpg`，從檔名取出像素座標，再正規化到 [-1, 1]。
  - `directory` 預設是 `data/dataset_xy_2026-10-21/dataset_xy`。
- `make_loaders(directory, general, perlin, compression, ..., split='random')`：回傳 train / val 的 DataLoader。
  - `split='random'`：用固定 seed 隨機切 9:1。
  - `split='session'`：讀資料集旁邊的 `metadata/`（每張圖一個同名 JSON，含 `session_id`、`captured_at`），**照拍攝段落切**。同一個 `session_id`（一次收集）裡，相隔超過 `segment_gap`（預設 60 秒）就算新的一段；驗證集一次拿整段，相鄰的相似照片就不會分別落進 train 和 val。沒有 metadata 的圖各自算一段；只有一段時會報錯並提示改用 random。
  - **只有訓練集做 augmentation**，驗證集一律用原圖。
- 前處理和課程一致：resize 224 → tensor → RGB 轉 BGR → ImageNet 正規化。這樣和車上 `camera.preprocess` 的結果完全相同（已驗證差 0）。
- 修掉課程版的 bug：原本就算 `random_hflips=False` 也會隨機翻轉。

### `preprocessing.py`

位在 data loader 和 augmentation 之間，**只做一件事：把車道線標出來**。跟 augmentation 的差別是：每次都做一樣的處理，而且車上推論時也要做。所以模型有沒有開前處理會存進 `.json`，`live_demo.ipynb` 載入模型時會自動套用同一套。開關是 `train.py --lanes`，不加就不做，舊模型不受影響。

- **做法**：把看起來像白膠帶的像素塗成純紅色 (BGR 0,0,255)。模型仍然看得到原本的畫面（3 通道，ResNet 不用改），只是車道線被明確標出來。標籤不變。
- **能處理彎道**：逐像素判斷，不是用直線去套。彎道、髮夾彎、鏡頭前面很寬的膠帶都標得出來。
  1. 地板亮度：取畫面下半部中間 LAB 亮度的第 40 百分位。
  2. 膠帶：比地板亮 `delta`（70）以上；或是細的亮條紋（white top-hat 21 px 大於 35，而且比地板亮 40 以上），用來抓遠處比較細、比較暗的膠帶。
  3. 去掉牆壁、場外木地板這類「又厚又在上面」的亮區（厚度超過 31 px，而且位在畫面上方 55% 或碰到上緣）。位置低的厚亮區是鏡頭前的膠帶，所以保留。
  4. 去掉小雜點，以及畫面上方 1/4 的小塊。
- 參數是用 224×224 的畫面調的，其他尺寸會等比例縮放 kernel。參數存在模型 `.json`，車上用同一組。
- **在預計訓練的 198 張上測過**：每張都有標出膠帶；沒有任何一張的標籤點（綠點）落在被標成紅色的地方；PC CPU 每張約 6 ms。Nano 上會更慢，`live_demo` 顯示的延遲已經包含前處理時間。
- 已知的不完美：畫面最上方的人、腳、椅子偶爾有小塊被標紅；場外木地板的紋路偶爾有雜點。
- `Preprocessor(lanes)(image, x, y)` 訓練時用；`apply_frame(frame)` 車上用。兩邊走同一段程式，餵進模型的 tensor 完全相同（已驗證差 0）。
- 順序是先標車道線、再做 augmentation，所以顏色擾動、柏林噪聲也會套到紅線上，模型會學到紅線不一定是完全一樣的紅色。

**預覽工具**（PC）：

```bash
python preprocessing.py review --data <資料夾>                         # 全部畫上車道線標示和標籤
python preprocessing.py review --data <資料夾> --model models/X.pth    # 用模型的前處理，依誤差排序，最差的在前面
```

會產生 `label_review.jpg`：紅色是標出的車道線，綠點是標籤，青色圈是模型的預測。終端機也會依同樣的順序列出檔名。

### `augmentation.py`

`Augmenter(general=False, perlin=False, compression=False)`，呼叫方式是 `aug(image, x, y) -> (image, x, y)`。三個開關**全關時原樣輸出**。

| 開關 | 內容 |
|---|---|
| `general` | 水平翻轉、±10° 旋轉、80~100% 裁切再放大、亮度 / 對比 / 飽和度 ±0.3。幾何變換會同步移動標籤座標 |
| `perlin` | 柏林噪聲做出大片明暗不均，模擬光線和陰影 |
| `compression` | JPEG 低品質（q 5~35）重壓，或像 YouTube 144p 那樣縮小、重壓再放大 |

### `model.py`

- `build_model()`：ImageNet 預訓練的 ResNet-18，`fc = Linear(512, 2)`（投影片 P.39、P.45）。
  - `arch` 參數是給之後換別的模型用的，目前只支援 `resnet18`。
- `rebuild_resnet18_from_state_dict(sd)`：依權重的形狀重建網路。剪枝後 channel 數變少，直接 `load_state_dict` 會 size mismatch。
- `load_model(path)`：自動判斷 `.pth` 是 PyTorch 權重還是 TensorRT 包裝。TRT 的輸入 dtype 從 engine 讀，所以 trtexec 建的 FP32 輸入不會被誤餵 half。
- `list_models(dir)`：給介面下拉選單用，顯示名稱來自同名的 `.json`。

### `compression.py`

- `prune_model(model, ratio)`：DepGraph 結構化剪枝。
  - 用 torch-pruning 的 `MetaPruner` + `GroupMagnitudeImportance(p=2)`，`fc` 不剪，`round_to=8`。
  - 有殘差連接的層會一起剪，剪完仍是合法的 ResNet-18，只是 channel 變少。
- `count_macs_params()`、`export_onnx()`：ONNX 用 opset 11，輸入輸出名稱是 `input_0` / `output_0`。
- `wrap_trt_engine(engine, out)`：把 `trtexec` 建好的 engine 包成 `TRTModule` 能讀的 `.pth`，並自動產生 label 加上「| TensorRT」的 `.json`。

| 剪枝比例 | MACs | Params |
|---|---|---|
| 不剪 | 1.82G | 11.18M |
| 30% | 0.84G | 5.27M |
| 50% | 0.49G | 2.80M |

### `camera.py`

- `open_camera(source)`：`'jetbot'`（CSI 相機）、`'csi'`（同一顆相機，但只保留最新畫面）、`'opencv'`（USB）、`'folder'`（輪播資料集圖片）、`'dummy'`（假畫面）。後兩種沒有車也能測。
- `flush()`：丟掉迴圈卡住期間堆積的舊畫面。換模型後會自動呼叫。
  - 原因：jetbot 的 GStreamer `appsink` 不會丟舊畫面。換模型時 TRT engine 反序列化會卡住 GIL，相機讀取執行緒跟著停，畫面就堆起來；之後讀取速度和相機速度一樣，堆積永遠消化不掉，畫面會一直延遲。
  - `'jetbot'` 來源的 flush 是重啟相機管線（約 1~2 秒）；`'csi'` 來源設了 `drop=true max-buffers=1`，本來就不會堆積。
- `preprocess(frame, device, dtype)`：投影片 P.59 的前處理，dtype 依模型決定，不再寫死 `.half()`。
- `degrade()`：在即時畫面上加柏林噪聲 / 壓縮失真，給介面的「輸入干擾」用。
- `draw_overlay()`、`to_jpeg()`：畫預測點和轉向條，供畫面顯示。

### `control.py`

依投影片 P.69、P.75 的 `execute()`：

```
x = xy[0];  y = (0.5 - xy[1]) / 2
angle    = arctan2(x, y)
steering = Kp·angle + Kd·(angle − angle_last) + bias
left  = clip(speed + steering, 0, 1)
right = clip(speed − steering, 0, 1)
```

同一個 steering 一邊加、一邊減，兩輪互相拮抗產生轉向：steering > 0 右轉，< 0 左轉。用法是 `PDController(speed, kp, kd, bias).update(x, y) -> (left, right)`，各項中間值放在 `.last`。

髮夾彎用的兩個擴充（投影片沒有，**預設關閉**；關閉時和上面的公式完全一樣）：

| 參數 | 作用 |
|---|---|
| `turn_slowdown` k（0~1） | 轉向越大越慢：`speed_eff = speed × max(0, 1 − k·|angle|)`，直線維持原速 |
| `allow_reverse` | 馬達下限從 0 改成 −1，內側輪可以反轉，車子能原地轉 |

### `train.py`

```bash
pip install -r requirements-pc.txt
python train.py --help
```

| 參數 | 說明 |
|---|---|
| `--lanes` | 前處理開關：標示車道線。不加 = 不做前處理 |
| `--general` `--perlin` `--compression` | augmentation 開關，全不加 = 不做 augmentation |
| `--epochs 70` `--lr 1e-3` `--batch-size 8` | baseline 訓練（預設值同課程） |
| `--prune 0.3 0.5` | 剪枝比例，可一次給多個 |
| `--finetune-epochs 15` `--finetune-lr 1e-4` | 剪枝後的 finetune（必做） |
| `--onnx` | 同時匯出 ONNX，給板子建 TensorRT engine |
| `--base X_base.pth` | 跳過 baseline 訓練，直接剪既有模型（前處理和 augmentation 設定沿用它的 .json，剪枝後的模型輸入一定和 baseline 相同） |
| `--split session` `--metadata DIR` `--segment-gap 60` | 照拍攝段落切 train / val（`--metadata` 預設是 `--data` 旁邊的 `metadata/`） |
| `--data` `--out` | 資料集路徑、輸出資料夾（預設 `models/`） |

產出：`models/resnet18_<aug>_{base,pruned30,pruned50}.{pth,json,onnx}`；有開前處理時是 `resnet18_lanes_<aug>_...`，例如 `resnet18_lanes_gen_base.pth`。`.json` 裡有 loss 曲線、MACs / params、資料切分，寫報告時用得到。

### `live_demo.ipynb`

| 區塊 | 內容 |
|---|---|
| 變因 | 模型下拉選單（augmentation × 剪枝 × PyTorch / TRT），以及「輸入干擾」：柏林噪聲、壓縮失真 |
| 畫面 | 模型實際看到的影像（用 `--lanes` 訓練的模型會看到紅色車道線）、綠色預測點、上方橘色轉向條 |
| 控制參數 | speed gain、steering gain (Kp)、steering kd、steering bias（投影片 P.71 的範圍） |
| 髮夾彎（預設關閉） | 轉彎減速 k、內側輪可反轉 |
| 即時讀數 | x、y、angle、steering、實際 speed、左右馬達（−1~1）、P / D 項、FPS、推論延遲 |
| 運轉 | 開始推論、**停止**、**允許移動**（預設關閉）、錄製 CSV |

---

## DEMO 前（PC / Colab，有 GPU）

```bash
cd project4
python train.py --prune 0.3 0.5 --onnx                                   # 無 augmentation
python train.py --general --prune 0.3 0.5 --onnx                         # 一般做法
python train.py --general --perlin --prune 0.3 0.5 --onnx                # + 柏林噪聲
python train.py --general --perlin --compression --prune 0.3 0.5 --onnx  # + 壓縮失真
```

看 log 裡的 val loss：如果剪枝 50% 的 finetune 後明顯比 baseline 差，DEMO 就用 30% 的版本。

加上前處理（標示車道線）的流程：

```bash
# 1. 先看車道線標得對不對
python preprocessing.py review --data <資料夾>
# 2. 有開、沒開各訓練一組，比 val loss 和行駛結果
python train.py --data <資料夾> --general --prune 0.3 0.5 --onnx
python train.py --data <資料夾> --lanes --general --prune 0.3 0.5 --onnx
```

## DEMO 前一天（Nano）：建 TensorRT engine

engine 必須在板子上建，每個模型要幾分鐘，**不要留到 DEMO 當天**。

1. 把以下檔案同步到板子的同一個資料夾（例如 `~/project4/`）：
   - `*.py`、`live_demo.ipynb`
   - `models/` 裡的 `.pth`、`.json`、`.onnx`
2. 在板子的終端機，對每個 `.onnx` 建 engine 並包裝：

   ```bash
   cd ~/project4/models
   for f in *.onnx; do
     n="${f%.onnx}"
     /usr/src/tensorrt/bin/trtexec --onnx="$f" --saveEngine="$n.engine" --fp16
   done
   cd ..
   python3 -c "
   import glob; from compression import wrap_trt_engine
   for e in glob.glob('models/*.engine'):
       print(wrap_trt_engine(e, e[:-len('.engine')] + '_trt.pth'))"
   ```

3. 開 `live_demo.ipynb`，執行到第 4 節，選一個 `| TensorRT` 的模型按「開始推論」，**先確認 TRT 能載入**。
   - 如果出錯，看「已知風險」那一節。PyTorch 的 `.pth` 一樣可以 DEMO。
4. 第 5 節的 `benchmark()` 可以先跑一次，記下各模型的延遲給報告用。

## DEMO 當天：操作車子

### 1. 開機連線

1. 車子開機，筆電連上板子的 AP，打開 Jupyter（port 8888）。
2. **把其他 notebook 的 kernel 都關掉**。相機同時只能被一個程式佔用。
   - 用 Running 分頁或 Kernel → Shut Down。

### 2. 啟動介面

1. 開 `live_demo.ipynb`，確認第 1 節的設定是 `CAMERA_SOURCE = 'jetbot'`、`USE_ROBOT = True`。
2. 由上往下執行第 1 到第 6 節，用 Run All 也可以。第 3 節會出現介面，第 6 節是「結束」按鈕，**不會自動執行**。

### 3. 架高測試（輪子離地）

1. 下拉選單選要 DEMO 的模型，按 **開始推論**。
2. 把車子放在跑道上不同位置，確認綠點落在該去的方向，FPS 和延遲合理。
3. 把車架高讓輪子離地，speed gain 調到 0.1，勾 **允許移動**。
   - 兩輪都會轉。目標在右邊時左輪比較快，目標在左邊時右輪比較快。
   - 方向反了就先停下來檢查接線或程式。
4. 按 **停止**。

### 4. 放到跑道上

照投影片 P.72 的原則：先用低速，一次只改一個參數。

1. 起始值：speed gain **0.10~0.15**、steering gain 0.2、kd 0、bias 0。
2. 把車放在直線段，勾 **允許移動**。
3. **先調 steering gain (Kp)**：轉彎轉不過去就加大；左右擺動就減小。
4. 擺動還是明顯時，**再加 kd**，從 0.01~0.05 慢慢加。太大會放大預測雜訊，反而抖。
5. 直線上一直往同一邊偏，就調 **bias**（-0.3~0.3），補償左右馬達的差異。
6. 穩定之後再慢慢提高 speed gain。速度變快通常要重新微調 Kp / Kd。
7. **壓到彎道內側**：通常是目標點太遠、切彎。先把 Kp 調小一點（例如 0.2 → 0.15）。
8. **髮夾彎轉不過去**：
   - 先開 **轉彎減速 k**，從 0.3 開始，逐步加到 0.6。轉彎時會自動變慢，直線不受影響。
   - 還是轉不過去，再勾 **內側輪可反轉**，讓車子可以原地轉。
   - 根本的解法是補拍髮夾彎的資料（入彎、彎中、出彎各角度 30~50 張，目標點放近），再重新訓練。

### 5. 比較不同變因

- 從下拉選單換模型時，車子會**自動停下並取消「允許移動」**。
  - 換完會自動清掉相機堆積的舊畫面（畫面會停 1~2 秒），之後畫面就是即時的。
  - 換完先看綠點正不正常，再手動勾回「允許移動」。
- 勾「輸入干擾」可以現場示範：有柏林噪聲 / 壓縮 augmentation 的模型在干擾下比較穩。
- 勾 **錄製 CSV**，每一幀的模型、x、y、steering、左右馬達、延遲、Kp / Kd / bias 都會寫進 `runs/drive_*.csv`，報告可以用來畫圖。

### 6. 停車與結束

- 隨時按 **停止**：立刻停車、取消允許移動、停止推論。
- 狀況不對時，直接把車拿起來再按停止。
  - Jupyter 的 interrupt（■）**停不了背景推論迴圈**，要用「停止」按鈕。
- 結束時按第 6 節的 **結束：停車並釋放相機** 按鈕。下次要開別的 notebook 前一定要按。
  - 按了之後相機就釋放了，要再用必須 Restart Kernel 再重新執行。

### 報告要留的證據（投影片 P.79、P.80）

- [ ] 各模型 `.json`：epoch 數、訓練 / 驗證張數、loss 曲線、MACs / params
- [ ] `benchmark()` 的延遲 / FPS，所有模型在相同條件下量
- [ ] `runs/drive_*.csv` 和行駛影片：跑幾圈、有沒有出界、擺動情況
- [ ] 最後用的 speed gain、Kp、Kd、bias，以及調整過程

## 疑難排解

| 狀況 | 處理 |
|---|---|
| 換模型之後畫面一直延遲 | 確認第 1 節 `FLUSH_ON_SWITCH = True`。還是延遲的話，改用 `CAMERA_SOURCE = 'csi'`（Restart Kernel 後重跑） |
| 按了開始推論，畫面卻沒更新 | **先看模型選單下方有沒有紅字**。推論迴圈出錯時會停下，並把錯誤和 traceback 顯示在那裡 |
| 相機開不起來 | 其他 notebook 還佔著相機：關掉所有 kernel 再試，還不行就 `sudo systemctl restart nvargus-daemon` |
| 綠點亂跳或完全不對 | 是模型或資料的問題，不是控制的問題（投影片 P.61）。換一個模型比較看看 |
| 綠點合理但車子擺動 | 是控制的問題：降低 Kp 或 speed gain，再加一點 Kd |
| 選 TRT 模型出錯 | 見下方「已知風險」，先改用同名的 PyTorch `.pth` |
| `No module named torch2trt` | 只有 TRT 模型需要 torch2trt。在 Nano 上確認有裝 |
| 介面卡住沒反應 | 按停止，重新執行第 4 節，再按開始推論 |

## 已知風險（沙盒內無法驗證）

- **TensorRT**：舊版 torch2trt 的 `TRTModule` 用 implicit batch 的 `execute_async`，碰到 trtexec 建的 explicit batch engine 可能報錯。
  - 備案是用 PyTorch `.pth` DEMO（CUDA 上自動 FP16）。
  - 或照投影片 P.55 用 `torch2trt` 轉：模型改用 `rebuild_resnet18_from_state_dict()` 建立。
- Nano 上的 Python 3.6 / 舊版 ipywidgets 只做過靜態語法檢查，介面排版可能和 PC 上不同。
- 換模型時的 flush（重啟 jetbot 相機管線）和 `CAMERA_SOURCE = 'csi'` 都還沒在板子上試過。flush 有問題就把 `FLUSH_ON_SWITCH` 改成 `False`。
- 和投影片流程不同的地方，報告要寫清楚，DEMO 前建議先問助教：
  - 訓練和剪枝在 PC 上做，不是在 JetBot 上（P.79）。
  - TensorRT 轉換走 ONNX → trtexec，不是 torch2trt（P.56）。
