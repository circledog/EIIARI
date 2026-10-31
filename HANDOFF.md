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
- 「允許移動」預設關閉；「停止」立刻停車並取消許可；切換模型時先停車並 reset 控制器。
- 「輸入干擾」：在即時畫面上加柏林噪聲／壓縮失真，測穩健性（額外加的功能）。
- 「錄製 CSV」寫到 `runs/drive_*.csv`；第 5 節 `benchmark()` 在相同條件下量所有模型的延遲，供報告使用。
- `CAMERA_SOURCE='dummy'` 或 `'folder'` 加上 `USE_ROBOT=False`，沒有 JetBot 也能跑。

## 未完成

- 用真實資料在 CPU 上跑 train.py 的 smoke test（含剪枝、ONNX）。
- 用 dummy 相機與假 robot 執行一次 live_demo.ipynb。
- 板子端（TRT、實際行駛）只能由使用者實測。

## 測試指令與結果

```bash
# Python 3.6 相容性（Nano 端模組）
vermin --no-tips -t=3.6- model.py control.py camera.py compression.py augmentation.py
# → Minimum required versions: 3.3  ✅
```

## 下一步

- sandbox 的 `download.pytorch.org` 被防火牆擋（需 host 審核），改從 PyPI 安裝 torch 中。
