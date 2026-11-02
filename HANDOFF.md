# HANDOFF — Project 4 pipeline (branch `feat/pipeline`)

Worktree: `/home/user/projects/EIIARI-pipeline`（從 `main` 開出）。所有程式在 `project4/`。

## 任務目標

把 Project 4（JetBot 道路跟隨）拆成模組，並做一個可切換變因的 live demo notebook：

```
Data Loader → Data Augmentation → Model training → Model compression      (PC)
Camera Input → Model inferencing → Motor controlling                       (Jetson Nano)
```

| 檔案 | 內容 |
|---|---|
| `data_loader.py` | `XYDataset`，路徑參數，預設 `data/dataset_xy_2026-10-21/dataset_xy` |
| `preprocessing.py` | ROI 裁切、固定 y、排除清單；設定存進模型 .json，車上也套用；`review` 標籤檢視工具 |
| `augmentation.py` | 三組開關：一般（幾何＋色彩）、柏林噪聲、壓縮失真（JPEG / 144p）。全關＝原樣輸出 |
| `model.py` | ResNet-18（ImageNet pretrained）+ `fc=Linear(512,2)`；依 state_dict 重建剪枝後網路；TRT 載入 |
| `compression.py` | DepGraph 剪枝（torch-pruning `MetaPruner` + `GroupMagnitudeImportance(p=2)`）、MACs/params、ONNX 匯出、engine 包裝 |
| `camera.py` | 相機輸入（jetbot.Camera / OpenCV / 假畫面）與前處理 |
| `control.py` | 投影片 P.69–75 的 PD + bias 差速（拮抗）控制 |
| `train.py` | PC 訓練入口：訓練 baseline → 剪枝 → finetune → 匯出 |
| `live_demo.ipynb` | Nano 上推論＋控制，UI 切換模型、即時畫面、馬達參數 |

## 重要決策

- 「DCP」依使用者提供的前次交接文件解讀為 **DepGraph 結構化剪枝**（torch-pruning），不是 Discrimination-aware Channel Pruning。
- 模型先用投影片的 ResNet-18；「other model」尚未決定。`model.py` 用 `build_model(arch=...)` 預留擴充點。
- augmentation 是訓練期變因 → live demo 的「切換變因」＝切換不同條件訓練出的 checkpoint。
- 前處理沿用課程慣例：相機給 BGR，訓練時 `[::-1]` 把 RGB 轉 BGR，兩邊都是 BGR 進網路。
- 修正課程 `XYDataset` 不看 `random_hflips` 的 bug；驗證集不做任何 augmentation。
- 資料集只有 jpg、沒有 metadata → train/val 隨機 9:1（固定 seed）。
- Nano 端（JetPack 4.x / Python 3.6，版本未確認）會 import 的模組保持 Python 3.6 相容：不用 dataclass、walrus、`from __future__ import annotations`、`list[int]` 型別。
- `project4/data` 在 worktree 裡是指向 `/run/sandbox/source/project4/data` 的 symlink（gitignore 已改為 `/data` 才能涵蓋 symlink）。

## 已完成

- 讀完投影片（84 頁）與前次交接文件，建立 worktree。
- 寫完全部模組：`data_loader.py`、`augmentation.py`、`model.py`、`compression.py`、`camera.py`、`control.py`、`train.py`、`live_demo.ipynb`。
- 控制依投影片 P.75 `execute()`：`y=(0.5-xy[1])/2`、`angle=atan2(x,y)`、`steering=Kp·angle+Kd·Δangle+bias`、左 `speed+steering`／右 `speed-steering`，clip 到 [0,1]。
- augmentation 驗證：幾何變換後標籤誤差平均 0.65 px（最大 2.1 px，來自點的取樣）；全關時回傳同一個物件。

## live_demo.ipynb 功能

- 下拉選單切換 `models/*.pth`（顯示 `.json` 裡的 label、params、MACs、val loss）；PyTorch 和 TRT `.pth` 自動判斷，TRT 的輸入 dtype 從 engine 讀。
- 即時畫面＋預測點＋steering 條；x、y、angle、steering、左右馬達、P、D、FPS、推論延遲。
- 投影片四個滑桿（speed gain、steering gain、kd、bias）。
- 「允許移動」預設關閉；「停止」立刻停車並取消許可；切換模型時停車、**取消允許移動**、reset 控制器。
- 「輸入干擾」：在即時畫面上加柏林噪聲／壓縮失真，測穩健性（額外加的功能）。
- 「錄製 CSV」寫到 `runs/drive_*.csv`；第 5 節 `benchmark()` 在相同條件下量所有模型的延遲，供報告使用。
- `CAMERA_SOURCE='dummy'` 或 `'folder'` 加上 `USE_ROBOT=False`，沒有 JetBot 也能跑。

## 使用流程

PC / Colab（`pip install -r requirements-pc.txt`）：

```bash
cd project4
python train.py --prune 0.3 0.5 --onnx                                   # 無 augmentation
python train.py --general --prune 0.3 0.5 --onnx                         # 一般做法
python train.py --general --perlin --prune 0.3 0.5 --onnx                # + 柏林噪聲
python train.py --general --perlin --compression --prune 0.3 0.5 --onnx  # + 壓縮失真
# 產出 models/resnet18_<aug>_{base,pruned30,pruned50}.{pth,json,onnx}
```

Nano：把 `models/*.pth`、`*.json`、`*.onnx` 和 `project4/*.py`、`live_demo.ipynb` 同步到板子後：

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/X.onnx --saveEngine=models/X.engine --fp16
python3 -c "from compression import wrap_trt_engine as w; w('models/X.engine','models/X_trt.pth')"
# wrap_trt_engine 會自動把 models/X.json 複製成 X_trt.json，label 加上 ' | TensorRT'
```

完整的車子操作步驟（DEMO 前一天建 engine、當天架高測試、調參順序、疑難排解）寫在 `project4/README.md`。

然後開 `live_demo.ipynb` 由上往下執行。`models/` 裡 PyTorch 和 TRT 的 `.pth` 會一起出現在下拉選單。

## 第二階段（README 與 DEMO 操作）

- `project4/README.md` 改寫：各模組簡介、PC 訓練指令、DEMO 前一天建 TRT engine、DEMO 當天操作步驟、調參順序（P.72）、報告證據清單、疑難排解。
- `compression.wrap_trt_engine` 會自動產生 TRT 版的 `.json`（label 加上 `| TensorRT`），讓下拉選單分得出 PyTorch 版和 TRT 版。用假 engine 測過 ✅
- live_demo：切換模型時自動取消「允許移動」，避免新模型一載入完車子就自己開走。nbclient 測過：切換前左馬達 0.546 → 切換後 0 / 0、`allow_move=False` ✅
- notebook 由 scratchpad 裡的 `make_nb.py` 產生（沒有進 repo）；之後改 notebook 可以直接在 Jupyter 裡改。

## 第三階段（板子實測回報的修正，2026-11-02）

使用者在 Nano 上實測時遇到的問題與修正：

- `TypeError: unexpected keyword argument 'weights'`：Nano 上是舊版 torchvision（< 0.13），`_resnet18(False)` 呼叫 `resnet18(weights=None)` 時丟出 `TypeError`，原本只接 `AttributeError`。已改成 `except (AttributeError, TypeError)`。`_trt_input_dtype` 不動。
  - 已測：模擬舊版的兩種失敗情況（拒絕 `weights=`、沒有 `ResNet18_Weights`），都能退回用 `pretrained=` ✅
- 畫面沒更新：真正原因是上面的 `TypeError` 讓推論迴圈在載入模型時就結束了。現在迴圈出錯時，會把錯誤和完整 traceback 顯示在模型選單下方，並把 `state['model']` 清空，修好後可以直接重按開始。
- Run All 會跑到「6. 結束」並關掉相機：改成按鈕（照使用者提供的程式碼），另外加上 `state['closed']`，按過結束之後再按開始會提示要 Restart Kernel。
- `camera.py` 維持原版（沒有 `start()`）。板子上曾經加過、後來又還原的 `start()` 和 preview 執行緒都不是原因，repo 裡沒有加。
- 已測（nbclient，模擬 Run All，連結束按鈕的 cell 一起執行）：相機沒被關掉、推論正常；選壞掉的模型時紅字和 traceback 有顯示；換回正常模型後可以重新開始；按結束後再按開始會被拒絕 ✅

## 第四階段（髮夾彎控制、換模型後畫面延遲，2026-11-02）

- 使用者回報：車子會壓到彎道內側（Kp 0.2、Kd 0.01）；資料集只有直道和一般彎道，沒有髮夾彎。
  - 抽樣檢查標籤：急彎 / 雙線轉角處，有些標籤點靠近粗線；x 分布兩端很少（`[3,23,24,50,48,28,20,4]`），缺少快壓線時往回拉的畫面；y 從 72 到 212 都有，標記遠近不一致。
- `control.py` 新增兩個選項，預設關閉，關閉時和投影片公式一模一樣：
  - `turn_slowdown`：`speed_eff = speed·max(0, 1 − k·|angle|)`。
  - `allow_reverse`：馬達下限改成 −1。
  - live_demo 加了「轉彎減速 k」滑桿、「內側輪可反轉」開關，左右馬達讀數改成 −1~1，另外顯示實際 speed。CSV 多了 `speed_eff`、`turn_slowdown`、`reverse` 三欄。
- 換模型後畫面持續延遲：jetbot 的 `appsink` 沒有設 `drop`，載入模型時 GIL 被佔住，相機執行緒停下、畫面堆積，之後永遠消化不掉。修法：
  - 每個相機來源都加了 `flush()`。jetbot 來源用 stop + start 重啟管線，並等到新畫面進來；opencv 來源設 `BUFFERSIZE=1`，再一直 grab 到真正需要等待為止。
  - 換模型的順序改成：載入 → warmup 3 次 → flush → 開始使用。由 `FLUSH_ON_SWITCH` 控制。
  - 新增 `'csi'` 來源：自己建 GStreamer 管線，`appsink drop=true max-buffers=1 sync=false`。**沒有在實機上測過**。
  - 畫面和讀數最多每秒更新 `DISPLAY_FPS=15` 次（控制仍然每幀都做），避免 Wi-Fi 送不完、在瀏覽器端排隊。
- 已測（假的 jetbot 模組，`fakejet/`：appsink 式的無上限佇列，讀取速度等於相機速度，載入時讓相機停 2 秒）：
  - 不 flush：換模型前畫面落後 2.07 秒（第一次載入），換模型後 4.03 秒，而且會累加。
  - flush：0.08 秒 → 0.09 秒 ✅
  - 新的控制選項有正確傳到控制器；vermin 檢查 3.6 相容 ✅
- 沒有實機驗證：jetbot `Camera.stop()` / `start()` 重啟在 Nano 上是否穩定、要多久；`'csi'` 管線字串（照 jetbot 0.4 的預設：3280×2464、sensor-mode 3）。

## 第五階段（preprocessing.py，2026-11-02）

使用者要求：在 data loader 和 augmentation 之間加一個 `preprocessing.py`，並更新 ipynb 和文件。

- 分類：標籤清理、ROI、固定 y 屬於前處理，這次已實作。合成偏離畫面（隨機）屬於 augmentation，Huber loss 是訓練設定，推論平滑屬於控制，這三項都**還沒做**。
- `Preprocessor(roi_top=0, fixed_y=None, exclude=None)`：三個開關全關時不做任何處理。
  - 訓練：`__call__(PIL, x, y)`。順序是先 fixed_y，再裁切並換算標籤。
  - 車上：`apply_frame(frame)` 走同一段 PIL 程式，`to_full(xy)` 把預測點換算回完整畫面座標。
  - `config()` 存進 `.json` 的 `preprocessing` 欄位（含 `exclude_path`），`from_config()` 讀回來；舊模型沒有這個欄位＝不處理。
- `data_loader`：切分先在全部 200 張上做，再從兩邊拿掉排除的圖，所以有沒有排除，驗證集都是同一批圖。
- `train.py`：新增 `--roi-top`、`--fixed-y`、`--exclude`。檔名和 label 會帶上前處理的 tag。`--base` 會沿用 baseline 的前處理設定和排除清單（如果檔案還在）。
- `live_demo`：載入模型時建立對應的 Preprocessor，推論前套用；預測點換算回完整畫面後再給控制和顯示。被裁掉的部分變暗，並畫一條青色線。benchmark 也有套用前處理。
- `camera.draw_overlay(..., roi_top=0)` 新增畫出 ROI 的功能。
- 已測：
  - 標籤換算和 `to_full` 來回一致，誤差 0；fixed 0.65 + roi 0.3 → 標籤落在裁切後的正中間，換算回完整畫面是 0.3 ✅
  - 訓練端和車上端的輸入，同一張圖差 0 ✅
  - 排除清單 2 張：200 → 198 張，切分 179 / 19 ✅
  - `train.py --roi-top 0.3 --fixed-y 0.65 --exclude … --general --prune 0.3 --onnx` 3 epochs：val 0.0275 ✅；`--base` 沿用前處理和排除清單 ✅
  - `preprocessing.py review --model …` 產生依誤差排序的總覽圖 ✅
  - notebook（folder 相機）：新模型 roi 0.3 / fixed_y 0.65 有套用、畫面上方有變暗；換回舊模型時 roi 0 ✅；vermin 3.6 ✅

## 未完成 / 未驗證

- **TensorRT 路徑完全沒有測過**（sandbox 沒有 GPU／TensorRT）：`load_model` 讀 TRT `.pth` 的部分、`_trt_input_dtype`、`TRTModule` 跟 trtexec 建出的 explicit-batch engine 是否相容（前次交接列的最大風險）。
- **Nano 上沒有實際跑過**：jetbot.Camera / Robot、Python 3.6 環境（只用 vermin 靜態檢查過）、ipywidgets 舊版的介面排版。
- 正式訓練（70 epochs、各種 augmentation 組合）還沒跑，smoke test 只跑了 2 epochs。
- 「other model」尚未決定。
- 若 Nano 上的 torch 沒有 `torch.load(..., weights_only=...)` 會 fallback；舊 torchvision 的 `pretrained=` 已處理。

## 測試指令與結果

Python 環境：`/tmp/claude-1000/.../scratchpad/venv`（Python 3.12，torch 2.15 CPU from PyPI，torch-pruning 1.6.0）。

```bash
# 1. Python 3.6 相容性（Nano 端模組）
vermin --no-tips -t=3.6- model.py control.py camera.py compression.py augmentation.py
# → Minimum required versions: 3.3  ✅

# 2. 真實資料 smoke test（200 張 → 180 train / 20 val）
python train.py --general --perlin --compression --epochs 2 --prune 0.3 --finetune-epochs 1 --onnx --workers 2 --out <tmp>/models
# → base val 0.0867；剪枝 30%：MACs 1.82G→0.84G、params 11.18M→5.27M，finetune 1 epoch 後 val 0.0435  ✅ 約 50 秒
python train.py --base <tmp>/models/resnet18_gen+perlin+comp_base.pth --prune 0.5 --finetune-epochs 1
# → MACs 1.82G→0.49G、params 11.18M→2.80M；檔名、augmentation 沿用 baseline 的 .json  ✅

# 3. 一致性
# - load_model 用 shape 重建 base / pruned30 → 與 onnxruntime 最大差 ~1e-7，opset 11，
#   op 只有 Add/Conv/Flatten/Gemm/GlobalAveragePool/MaxPool/Relu  ✅
# - 訓練前處理 (data_loader) vs 相機前處理 (camera.preprocess) 同一張圖：差 0  ✅
# - 控制：目標在右 (x=0.5,y=0.25)→ 左 0.63 / 右 0.00（右轉）；目標在左 → 左 0 / 右 0.74  ✅
# - augmentation 幾何標籤誤差平均 0.65 px  ✅

# 4. live_demo.ipynb 用 nbclient 無頭執行（CAMERA_SOURCE='dummy'、USE_ROBOT=False）
# → 推論 ~33 FPS（CPU）；未勾允許移動時馬達 0、勾了才送值；切模型＋輸入干擾正常；
#   停止後馬達 0、允許移動取消；CSV 263 行；benchmark：pruned30 12 ms、base 43 ms（CPU）  ✅
```

## 下一步

1. 在 PC / Colab 用 GPU 跑正式訓練（四種 augmentation 組合 × base / pruned30 / pruned50）。
2. 板子上跑 trtexec → wrap → live_demo，先驗證 TRT 載入（風險最高）。
3. 驗證 Nano 上的介面與相機，再調 PD 參數。
4. 決定 other model 後在 `model.build_model(arch=...)` 和 `rebuild_*` 加上對應的分支。
