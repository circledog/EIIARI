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
| `preprocessing.py` | 只做車道線標示：白膠帶塗成紅色，彎道也適用；固定開啟、即時做、不存檔。訓練在 DataLoader 裡（augmentation 之後）呼叫 `mark_lanes_pil()`，車上用 `apply_frame()` |
| `augmentation.py` | 三組開關：一般（幾何＋色彩）、柏林噪聲、壓縮失真（JPEG / 144p）。全關＝原樣輸出 |
| `resnet18.py` | ResNet-18（ImageNet pretrained）+ `fc=Linear(512,2)`，加上它自己的 `train()` 和 `DEFAULTS` |
| `mobilenet_v2.py` | MobileNetV2（ImageNet pretrained）+ `classifier[1]=Linear(1280,2)`，加上它自己的 `train()` 和 `DEFAULTS` |
| `metrics.py` | MSE 和像素誤差（平均、x／y、p95） |
| `sanity.py` | 權重 SHA256、訓練時存的比對畫面與車上比對 |
| `convert_trt.py` | 車上把 `models/` 的模型轉成 TensorRT 放到 `models_trt/`，並比對、計時 |
| `loader.py` | 共用：`build_model(arch)`、`detect_arch`、依 state_dict 重建剪枝後網路、PyTorch／TRT 載入、`.json` |
| `compression.py` | DepGraph 剪枝（torch-pruning `MetaPruner` + `GroupMagnitudeImportance(p=2)`）、MACs/params、ONNX 匯出、engine 包裝 |
| `camera.py` | 相機輸入（jetbot.Camera / OpenCV / 假畫面）與前處理 |
| `control.py` | 轉向角（預設隊友公式，可切投影片公式）＋ PID ＋ 平滑＋差速控制（第十四階段） |
| `train.py` | PC 訓練入口：依任務呼叫各網路的 `train()`；baseline → 剪枝 → finetune → 匯出，可多 seed |
| `live_demo.ipynb` | Nano 上推論＋控制，只列 TensorRT 模型，「開始行駛／停止」按鈕、安全機制 |

## 重要決策

- 「DCP」依使用者提供的前次交接文件解讀為 **DepGraph 結構化剪枝**（torch-pruning），不是 Discrimination-aware Channel Pruning。
- 網路：投影片的 ResNet-18，加上 other model 的 MobileNetV2（第十一階段）。每個網路一個檔，共用功能在 `loader.py`；不再用 `model.py` 這個名稱，避免和 `models/` 資料夾搞混。
- augmentation 是訓練期變因 → live demo 的「切換變因」＝切換不同條件訓練出的 checkpoint。
- 前處理沿用課程慣例：相機給 BGR，訓練時 `[::-1]` 把 RGB 轉 BGR，兩邊都是 BGR 進網路。
- 修正課程 `XYDataset` 不看 `random_hflips` 的 bug；驗證集不做任何 augmentation。
- 現有資料集只有 jpg、沒有 metadata → 預設 train/val 隨機 9:1（固定 seed）。之後新拍的資料連 `metadata/` 一起複製，就可以用 `--split session` 照拍攝段落切（第六階段）。
- Nano 是 **Python 3.6.9**（原本收集 notebook 的 metadata 寫的），板子會 import 的模組保持 3.6 相容：不用 dataclass、walrus、`from __future__ import annotations`、`list[int]` 型別、`datetime.fromisoformat`。
- 前處理固定開啟、沒有開關、不存檔。訓練順序是「讀圖 → augmentation → 標車道線」，車上是「相機 → 輸入干擾 → 標車道線」，噪聲對偵測的影響兩邊一致（第十一階段）。
- 控制改用隊友的公式和控制器（第十四階段，取代第四階段「預設和投影片完全一樣」）；投影片公式保留成介面上的切換按鈕。
- 換模型後清空相機佇列：jetbot 來源用重啟管線，另外提供 `'csi'`（`drop=true`）來源（第四階段）。
- 資料收集不在這個分支的範圍：收集 notebook（隊友版與課程原始版）、`lane_label.py`、`使用說明.txt` 都已刪除（第八階段）。第六、七階段的紀錄保留作歷史，內容要看 git 歷史（`14f384d` 之前）。
- 前處理**只做車道線標示**（第九階段，取代第五、八階段的 ROI／固定 y／排除清單）：逐像素偵測白膠帶並塗成純紅色疊在原圖上，模型仍吃 3 通道彩色圖；不用直線擬合，所以彎道可以用。標籤（檔名的綠點）只用來檢查偵測結果，偵測本身不依賴它，因為車上推論時沒有標籤。
- **程式註解規則**（使用者要求，第十二階段）：每行有實質內容的程式碼，上一行都要有一條英文註解；不可寫在同一行後面。常見的 library import、單獨的括號／收尾、docstring 不加。多行的敘述只在第一行上方加一次。之後新增或修改程式都要照做。
- `project4/data` 在 worktree 裡是指向 `/run/sandbox/source/project4/data` 的 symlink（gitignore 已改為 `/data` 才能涵蓋 symlink）。

## 已完成

- 讀完投影片（84 頁）與前次交接文件，建立 worktree。
- 寫完全部模組：`data_loader.py`、`augmentation.py`、`model.py`、`compression.py`、`camera.py`、`control.py`、`train.py`、`live_demo.ipynb`。
- 控制依投影片 P.75 `execute()`：`y=(0.5-xy[1])/2`、`angle=atan2(x,y)`、`steering=Kp·angle+Kd·Δangle+bias`、左 `speed+steering`／右 `speed-steering`，clip 到 [0,1]。
- augmentation 驗證：幾何變換後標籤誤差平均 0.65 px（最大 2.1 px，來自點的取樣）；全關時回傳同一個物件。
- 之後的修改依序寫在下方「第二階段」到「第七階段」。

## live_demo.ipynb 功能（目前版本，第十四階段）

- 下拉選單只列 `models_trt/` 的 TensorRT 模型，一開始「未載入模型」，選了才載入；載入完成就開始推論，停車時也繼續推論。
- 載入時跑 sanity 比對（綠＝通過、紅＝沒通過→禁止開車、橘＝舊模型沒有比對資料）。`.json` 有 SHA256 時先比對，不對就拒絕載入。
- 「開始行駛」單一按鈕：綠色「開始行駛」↔ 紅色「停止」；安全機制停車時自動變回綠色並顯示原因。「開始推論」按鈕已移除。
- 安全機制（開車時）：輸出 NaN／超出畫面／目標太近、相機沒有新畫面、看門狗逾時（滑桿可調，預設 0.5 秒）、推論錯誤（停車並卸載模型）、切換模型。
- 控制選項全部是按鈕或滑桿：轉向角公式、max／min speed、kp、ki、kd、bias、轉彎減速、角度平滑、馬達變化率、馬達上限、內側輪可反轉、安全逾時、輸入干擾、錄製 CSV。
- 偵測不到 JetBot 就用假馬達；鏡頭打不開就顯示「未連接鏡頭」。沒有車也能看介面。
- 第 5 節 `benchmark()` 會先停車、暫停迴圈，量完自動恢復。第 6 節「結束」按鈕釋放相機。

## 使用流程

PC / Colab（`pip install -r requirements-pc.txt`）：

```bash
cd project4
python train.py --data data/splits --general --perlin --prune 0.3 0.5           # 產出 models/<網路>_<aug>_{base,pruned30,pruned50}.{pth,json}
python train.py --data data/splits --arch mobilenet_v2 --general --prune 0.3
python train.py --data data/splits --general --seeds 0 1 2 --prune 0.3          # 多 seed：平均 ± 標準差
python preprocessing.py review --data <資料夾>                                  # 前處理後的圖 + 標籤總覽
```

Nano：把 `project4/*.py`、`live_demo.ipynb` 同步到板子，PC 的 `models/*.pth` 和同名 `.json` 放到板子的 `models/`，然後：

```bash
python3 convert_trt.py          # 轉成 TensorRT 放到 models_trt/，每個模型顯示 PASSED / FAILED
```

再開 `live_demo.ipynb` 由上往下執行。

完整的車子操作步驟（DEMO 前一天建 engine、當天架高測試、調參順序、疑難排解）都寫在 `project4/README.md`。

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

## 第五階段（preprocessing.py，2026-11-02；ROI／固定 y／排除清單已被第九階段取代）

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

## 第六階段（資料收集檢查與修正，2026-11-02）

檢查 `project4_data_collection_gamepad.ipynb` 的發現：
- 200 張中只有 2 張 y=145（自動標記），**198 張是手動點的**，y 從 72 到 212，沒有統一規則。
- 自動標記（`detect_indoor` + 65% 高度取中點）在舊資料上只有 74/200 有效，跟手動標籤 x 的差距中位數 14 px。
- 只能直線前進，偏離／回正的畫面要手動擺；搬資料時 `metadata/` 沒有一起複製；只擋同一幀，不擋近似重複；手動點選用 widget 設定寬度換算座標（瀏覽器縮圖時會偏）；相機還沒出影格時直接報錯。

做了什麼：
- `lane_label.py`：把偵測程式從 notebook 搬出來，演算法**維持原樣**；`compute_midpoint()` 取代 `draw_lane_midpoint` 的計算部分；新增 `draw_guides()`；`python lane_label.py eval [--sheet]` 離線評估。
  - **試過但沒採用**：(a) 從畫面底部往上追蹤膠帶，取代用畫面中線分左右 → 72 vs 74，沒有改善；(b) 只有一條線時用「半個車道寬」推估 → 彎道會指錯方向，誤差中位數 60 px。原因是彎道前方膠帶接近水平、內側膠帶在畫面外，在固定高度上本來就沒有中點。結論：彎道要靠手動點選加上固定規則。
- 收集 notebook（「Provided setup cell. Do not modify.」那兩格沒有動）：
  - 手動點選**鎖定 y 在 far_y=0.65**（`MANUAL_LOCK_Y`，介面有勾選框），`label_source='manual_click_y_locked'`，metadata 多記 `y_locked`。
  - 左圖顯示 65% 參考線，只畫在顯示圖上；存檔用的 `raw_jpeg` 不變（已測：等於原 frame 的 JPEG）。
  - 點選座標用 click event 的 `width`/`height`（widget JS 會送出實際顯示大小）換算，沒有的話退回 widget 寬度。
  - 「標籤 x 分布」HTML 長條圖（`X_BINS=8`），存檔後更新並標出剛存的那一格。
  - 近似重複：32×32 灰階平均差小於 `NEAR_DUP_THRESHOLD=3.0` 時先擋，同一幀再按一次才存；重開 kernel 會讀最新一張存檔來比較。
  - 相機沒有影格時等待，最多 10 秒。
  - markdown 說明、拍攝建議、搬資料要連 metadata 一起，都更新了。
- `data_loader.split_by_session()` / `make_loaders(split='session')` / `train.py --split session --metadata --segment-gap`：照 session 和時間間隔分段，整段放進 val；一定會留資料給 train；只有一段時報錯。切分資訊寫進模型 .json 的 `split`。
- README（新增「資料收集」小節、`--split`）、`使用說明.txt`（開頭新增「〇、這次更新」）。

已測：
- `lane_label.py eval`：74/200、差距中位數 14 px，和原 notebook 的程式結果完全相同 ✅
- 收集 notebook 用假的 jetbot / ClickableImageWidget 透過 nbclient 執行（測試腳本 `collect_test.py`，假模組在 scratchpad 的 `fakecollect/`）：
  - 等待相機 → 恢復 ✅；顯示圖 ≠ 存檔圖、存檔圖＝原 frame ✅；自動存檔 ✅
  - 近似重複第一次擋、第二次存，換場景直接存 ✅
  - 顯示為 112 px 時點 (30,20) → x=60、y 鎖在 145 ✅；舊版 widget 沒有 width → 用 224 換算 ✅
  - metadata 欄位正確、`_thumb` 不會寫進 JSON、x 分布會更新 ✅
- session 切分：200 張模擬 33 段 / 3 個 session / 5 張沒有 metadata → 179/21，沒有任何一段跨兩邊 ✅；notebook 格式的 metadata 讀得懂；只有一段時給出清楚的錯誤 ✅；`train.py --split session` ✅；預設 random 沒有改壞 ✅
- vermin：收集 notebook 的程式碼 3.5、其他模組 3.x ✅（板子是 Python 3.6.9，見 notebook metadata）

沒有實機驗證：收集 notebook 在 Nano 上的實際操作（手把、ClickableImageWidget 的版本是否會送 width）。

## 第七階段（手把自動連線與綁定，2026-11-02）

- 查了 ipywidgets 前端原始碼：`ControllerModel.initialize()` 在前端建立時就會開始 `wait_loop`（用 `requestAnimationFrame` 輪詢 `navigator.getGamepads()[index]`），不需要顯示 view；斷線時 `reset_gamepad()` 會回到 `wait_loop`。瀏覽器規定按過鍵才讀得到手把，這一步省不掉。
- 收集 notebook：
  - `GAMEPAD_INDEX=None` → 建立 0~3 號四個 `Controller`，介面用 `controller_box` 只顯示目前使用中的那一支。
  - `GamepadBinding(lane, controllers)` 改寫：監看每支的 `connected` / `buttons`，第一支「已連線且按鍵數量夠」的就自動綁定；使用中的那支斷線 → 解除綁定、停車、改用另一支已連線的（如果有）；按鍵列表變了 → 自動重綁；按鍵不足會提示；重新執行那一格時，已經連著的手把會直接綁定。每次綁定都會先 `stop_all()`（取消允許移動）。
  - 「綁定手把」按鈕改成「重新綁定手把」，會重新讀取按鍵編號。
  - markdown、`使用說明.txt`、README 都更新了。
- 已測（`gamepad_test.py`，在 kernel 裡用 `set_trait` 模擬前端送來的更新）：2 號連上 → 自動綁定 ✅；13 / 5 / 1 號鍵分別觸發拍照 / 前進 / 停止 ✅；斷線 → 解除綁定、停車 ✅；0 號重連（先送 connected 再送 buttons）→ 自動重綁並可拍照 ✅；只有 4 個按鍵 → 提示 ✅；重新建立時手把已連著 → 直接綁定 ✅；改按鍵編號後手動重綁 ✅。`collect_test.py` 回歸測試全過 ✅；vermin 3.5 ✅。
- 沒有實機驗證：Nano 上舊版 ipywidgets 的 Controller 行為（推測相同）、實體手把。

## 第八階段（刪除資料收集、前處理開關，2026-11-02；前處理旗標已被第九階段的 `--lanes` 取代）

使用者要求：「刪掉 data collection」（確認範圍：三個全刪）、「Preprocessing 做成可開關」（確認：訓練用旗標開關）。

- 刪除 `project4_data_collection_gamepad.ipynb`、`data_collection_gamepad.ipynb`（課程原始）、`lane_label.py`，以及只描述收集 notebook 的 `使用說明.txt`。README 的「資料收集」小節、檔案表、已知風險裡的相關內容一併刪除。
- 保留 `--split session`：它只讀 `dataset_xy` 旁邊的 `metadata/*.json`（`session_id`、`captured_at`），不依賴收集 notebook。
- `train.py`：`--roi-top [FRAC]`、`--fixed-y [FRAC]` 變成開關，不給數值時用 `preprocessing.ROI_TOP_DEFAULT=0.3` / `FIXED_Y_DEFAULT=0.65`；新增 `--no-preprocess` 強制全關（也蓋過 `--base` 沿用的設定；baseline 有 ROI 時會印警告）。舊的寫法 `--roi-top 0.3 --fixed-y 0.65` 照樣能用。
- 已測：無參數 → 全關；`--roi-top` → 0.3；`--roi-top 0.4 --fixed-y` → 0.4 / 0.65；`--fixed-y --general` → 0.65；`--roi-top --no-preprocess` → 全關 ✅；以 `--roi-top --fixed-y` 訓練 1 epoch 的 baseline，`--base` 沿用 roi30+fy65、加 `--no-preprocess` 則全關 ✅。

## 第九階段（前處理改成只做車道線標示，2026-11-02；`--lanes` 開關與 `Preprocessor` 類別已被第十階段取代）

使用者要求：「Preprocessing 只做標示車道線，要能處理彎道」，並提供預計訓練的資料（`data.tar.gz`：198 張 224×224 魚眼畫面，另有 16 張 `.ipynb_checkpoints` 不會被讀到）。資料檔名的 xy 就是「綠點」，可以當行進方向的參考，但推論時沒有。確認的決定：結果直接畫在原圖上給模型看；ROI、固定 y、排除清單全部拿掉。

- 資料觀察：134 張 y=112、34 張剛好 (112,112)、16 張 x 在邊緣 0／223（急彎）。白膠帶的 LAB 亮度約 190~240，地板約 60~140；畫面上方有白牆、人、場外的木地板。
- `preprocessing.py` 重寫：`lane_mask()`、`mark_lanes()`、`Preprocessor(lanes, params)`。演算法見檔頭 docstring：亮度門檻加 top-hat 細條紋，再去掉位置高的大片亮區（牆），最後去雜點。kernel 依寬度 /224 等比例縮放。設定是 `{"lanes": bool, "params": {...}}`；舊模型的 `roi_top`／`fixed_y` 會印警告、當作沒有前處理。
- `review` 改成預覽工具：畫出紅色車道線和綠色標籤，`--model` 時依誤差排序；移除 `--exclude`。
- `train.py`：只剩 `--lanes`；`--base` 一定沿用 baseline 的設定（剪枝後的輸入要和 baseline 一致）。`data_loader` 移除排除清單；`camera.draw_overlay` 移除 `roi_top`。
- `live_demo`（`make_nb.py` 重新產生）：`frame = prep.apply_frame(frame)` 後推論並顯示，拿掉 `to_full`；model_info 顯示「前處理：標示車道線」；延遲改稱「前處理+推論」。
- 調整過程：原本的 top-hat 單一 kernel 會在鏡頭前寬膠帶中間挖洞，也會把牆壁和地板交界標進去 → 加亮度門檻；把「厚區塊」一律移除時，會連鏡頭前的寬膠帶一起刪掉 → 只刪位置高的；門檻 floor+45 會把灰色牆腳標進去 → 改成 +70；3×3 opening 會把遠處 1~2 px 的細線刪掉 → 細條紋不做 opening。
- 已測（scratchpad venv）：
  - 198 張：每張都有標出膠帶，0 張的標籤點落在紅色像素上，平均 5.8 ms／張 ✅；逐張看過總覽圖，彎道和髮夾彎都有標出來；剩下的誤判是畫面頂端的人和椅子、場外木地板的雜點。
  - 訓練路徑（PIL RGB → `Preprocessor` → `image_to_tensor`）和車上路徑（BGR → `apply_frame` → `camera.preprocess`）每隔 9 張比一次：最大差 0 ✅
  - `train.py --data <upload> --lanes --general --epochs 2 --prune 0.3 --finetune-epochs 1`：178/20、檔名 `resnet18_lanes_gen_*`、label 是 `pre=lanes`、`.json` 有參數 ✅；`--base` 沿用 lanes ✅
  - `live_demo` 用 nbclient 跑（folder 來源＝新資料、USE_ROBOT=False）：兩個 lanes 模型可以切換，model_info 顯示前處理，顯示畫面有紅色車道線（2.4% 像素），前處理+推論 19~28 ms（CPU），benchmark 正常 ✅
  - `preprocessing.py review`（有無 `--model`）✅；vermin `preprocessing.py camera.py data_loader.py` → 3.3 ✅
- 沒有實機驗證：Nano 上的前處理速度（OpenCV morphology 大約多 10~30 ms，推估）、實際相機畫面（未經 JPEG 壓縮）和訓練圖的差異。

## 第十階段（前處理固定開啟、輸出存成資料夾，2026-11-02；存檔與 `--preprocessed` 已被第十一階段取消）

使用者要求：「Preprocessing 的輸出在哪個資料夾？取消開關，但還是要留前處理過後的檔案。前處理做成函式給 train 叫，直接寫死在裡面，不用開關。」（第九階段的前處理是在 DataLoader 裡即時做的，沒有輸出檔案。）

- `preprocessing.py`：拿掉 `Preprocessor` 類別，改成函式：
  - `preprocess_dataset(src_dir, out_dir=None)`：把每張 `xy_*.jpg` 標好車道線後，存成同名的 PNG 到 `<src_dir>_lanes/`（無損），並寫入 `preprocessing.json`（參數、來源、張數）。已處理過的跳過；原圖不見的會刪掉；參數改了全部重做。
  - `apply_frame(frame)`：車上用，固定參數。
  - `CONFIG`：寫進模型 `.json` 的 `preprocessing`（加上 `dir`）。
  - CLI：`run`（只產生資料夾）、`review`（讀前處理後的資料夾）。
- 輸出放在原資料夾旁邊，這樣 `metadata/` 還是在同一層。`train.py` 的 `--split session` 預設改從 `--data` 找 metadata，所以 `--preprocessed` 指到別處也不受影響。
- `train.py`：拿掉 `--lanes`，新增 `--preprocessed DIR`（輸出位置，不是開關）。一開始固定呼叫 `preprocess_dataset()`，再用輸出資料夾建 DataLoader。檔名回到 `resnet18_<aug>_*`，label 拿掉 `pre=`。
- `data_loader.py`：拿掉 `preprocessor` 參數；`XYDataset` 同時讀 `xy_*.jpg` 和 `xy_*.png`。
- `live_demo`（`make_nb.py` 重新產生）：`from preprocessing import apply_frame as mark_lanes`，每幀固定套用；模型 `.json` 的 `preprocessing.method` 不是 `lane_marking` 時顯示紅字警告。
- 已測：
  - `train.py --data <copy>/dataset_xy --general --epochs 2 --prune 0.3`：產生 `dataset_xy_lanes/`（198 張 PNG＋`preprocessing.json`）；178/20；val loss 和第九階段完全相同（0.14322），表示處理結果一樣 ✅
  - 再跑一次：0 張新處理 ✅；刪掉 1 張原圖再跑：刪掉 1 張 PNG ✅
  - PNG（訓練）和 `apply_frame`（車上）餵進模型的 tensor 最大差 0 ✅
  - `live_demo` 用 nbclient 跑：可以切換模型、畫面有紅色車道線、benchmark 正常 ✅；`review --model` ✅；vermin 3.3 ✅

## 第十一階段（前處理改回即時、新增 MobileNetV2、拆分模型檔，2026-11-02）

使用者要求：
- 前處理不另存，改成即時標車道線。順序確認：先 augmentation 再標線，和車上一致。
- 測試腳本不保留，會在車上實測。
- 新增 other model；檔案名稱要區分，不要再用 `model`。確認：每個模型一個檔。

改動：
- `preprocessing.py`
  - 拿掉 `preprocess_dataset`、`default_output_dir` 和 `run` 子命令。
  - 新增 `mark_lanes_pil()`，給 DataLoader 用。
  - `CONFIG` 多一欄 `order: after_augmentation`。
  - `review` 直接讀原圖、即時標線。
- `data_loader.py`：`XYDataset.load()` 的順序是「讀圖 → augmenter → `mark_lanes_pil`」。驗證集也會標線，但不做 augmentation。只讀 `xy_*.jpg`。
- `model.py` 刪除，拆成三個檔：
  - `resnet18.py`、`mobilenet_v2.py`：各自只有 `NAME`、`FILE_TAG`、`LABEL`、`HEAD_KEY`、`build()`。
  - `loader.py`：`ARCHS`、`build_model`、`detect_arch`、通用的 `rebuild_from_state_dict`（depthwise conv 剪枝後 `groups` 跟著 channel 數）、`load_model`、`list_models`、`read_meta` / `write_meta`。
  - `compression.py` 改成 import `loader`。
- `train.py`
  - 新增 `--arch resnet18|mobilenet_v2`；`--base` 用 `detect_arch` 判斷網路。
  - 拿掉 `--preprocessed`。
  - 輸出檔名改成 `<FILE_TAG>_<aug>_*`（`resnet18_…`、`mobilenetv2_…`），label 改成 `<LABEL> | aug=… | …`。
- `live_demo.ipynb`（直接修改）：import 改成 `from loader import …`，流程說明同步更新。原本的順序已經是「`degrade` → `mark_lanes` → 推論」，不用改。

已測：
- 兩種網路都做了：建立 → 剪枝 30% → 從 state_dict 重建（差 0）→ 匯出 ONNX，和 onnxruntime 比對差 ≤1.2e-7 ✅。`detect_arch` 判斷正確 ✅。
  - ResNet-18：1.82G / 11.18M → 0.84G / 5.27M。
  - MobileNetV2：0.32G / 2.23M → 0.15G / 1.08M。
- `train.py` 加 `--general --perlin --epochs 2 --prune 0.3 --onnx --no-pretrained`，兩種網路都跑完 ✅。另外用 `--base mobilenetv2_..._base.pth --prune 0.5` 剪枝（0.09G / 0.58M）✅。
  - sandbox 連不到 download.pytorch.org，所以沒有測 MobileNetV2 下載預訓練權重。
- 訓練路徑（PIL → mark_lanes_pil → image_to_tensor）和車上路徑（BGR → apply_frame → camera.preprocess）餵進模型的 tensor 最大差 0 ✅。
- `live_demo` 用 nbclient 跑：5 個模型都能載入和切換，畫面有紅線，benchmark 正常 ✅。PC CPU 延遲：MobileNetV2 未剪枝 21.4 ms、剪枝 30% 13.7 ms；ResNet-18 未剪枝 21.4 ms、剪枝 30% 15.9 ms。
- `review --model` ✅；vermin 檢查 loader / resnet18 / mobilenet_v2 / preprocessing / data_loader / camera / compression → 3.3 ✅。

沒有實機驗證：
- MobileNetV2 在 Nano 上走 TensorRT。ReLU6 會匯出成 Clip，TensorRT 有支援。
- 板子上舊版 torchvision 有沒有 `mobilenet_v2`（0.3 版以後才有，JetPack 的版本應該有）。

## 第十二階段（每行加英文註解，2026-11-02）

使用者要求的規則：
- 註解放在所描述那行程式碼的**上一行**，用**英文**。
- 略過：常見的 library import、沒有實質內容的行（單獨的括號、分號、「end of dict」這類收尾）、docstring。

使用者以為之前提過這個要求，但對話紀錄和專案記憶裡都找不到；這次已寫進專案記憶（`line-comment-rule`）。

- 套用範圍：`project4/` 的 10 個 `.py`（augmentation、camera、compression、control、data_loader、loader、mobilenet_v2、preprocessing、resnet18、train），以及 `live_demo.ipynb` 的程式碼格。`train_model.ipynb` 是課程原始範例，沒有動。
- 處理方式：
  - 原本寫在同一行後面的註解，移到上一行。
  - notebook 原本的中文註解改成英文。介面上的中文字串和 markdown 說明不變。
  - `else:`、`except`、`finally:`、裝飾器、`def`、`class` 也都有註解。
  - 本專案自己的模組 import（例如 `from loader import …`）算實質內容，有加註解。
- 驗證：寫了檢查腳本，兩項都通過 ✅
  - 每個檔案、每個 notebook 程式碼格，加註解前後的 AST 完全相同，所以程式邏輯沒有任何改變。
  - 逐行檢查規則：每個需要註解的敘述上一行都是註解，而且沒有寫在同一行後面的註解；結果 0 個問題。
- 另外確認過：註解裡沒有中文；`train.py --arch mobilenet_v2 … --prune 0.3` 能跑完；`live_demo` 用 nbclient 跑正常。
- 順手修正：`compression.wrap_trt_engine` 的 docstring 原本寫 `model.load_model`，改成 `loader.load_model`。
- vermin 對 `train.py` 回報「`compression` 需要 3.14」，這是誤判：Python 3.14 新增了標準庫 `compression`，和本專案的 `compression.py` 同名。腳本所在的資料夾優先，所以實際 import 的是專案的版本；註解前就是這樣。

## 第十三階段（照隊友切好的 train / validation / test 資料夾訓練，2026-11-02）

使用者給了隊友切好的資料 `splits.rar`，內容是 `train/` 277 張、`validation/` 60 張、`test/` 61 張，合計 398 張：原本的 200 張加上新拍的 200 張，比例約 70/15/15。使用者確認的決定：
- 偵測到子資料夾就**自動**照資料夾用，不另外加旗標。
- `test` 只在最後評一次，寫進 `.json`。
- 壞檔跳過並警告。壓縮檔裡有 17 張解不開（train 14、validation 3），看起來是上傳不完整，解出來都是 0 byte。

改動：
- `data_loader.py`
  - 新增 `has_split_folders(dir)`：判斷是否同時有 `train/` 和 `validation/`。
  - `make_loaders` 改成回傳 `(train, val, test, augmenter)`。有子資料夾時照資料夾用（`split_info = {"split": "folders", 各資料夾張數, "skipped_unreadable"}`）；只有單一資料夾時和以前一樣用 random 或 session 切，test 是 `None`。
  - `XYDataset` 先完整解碼每張圖，讀不了的跳過，每個檔案只警告一次（`n_skipped`）。
- `train.py`
  - 新增 `report_test()`：base 和每個剪枝模型在 finetune 完、存檔前各評一次 test loss，印出來並寫進 `.json` 的 `test_loss`（沒有 test 時是 `null`）。用 `--base` 時也會印出 baseline 的 test loss。
  - 新增 `n_test`。有子資料夾又指定了 `--split` 或 `--val-ratio` 時，會印出提示說這兩個旗標被忽略。
- `live_demo.ipynb`：模型資訊列多顯示 `test loss`。
- README 已更新。

注意：新拍的那批是連拍，有 14 組相鄰的連續畫面（前後不到 2 秒）被分到不同資料夾，val 和 test 會稍微偏樂觀。使用者同意照隊友的分法用，不重切。

已測（sandbox，CPU）：
- `train.py --data splits --arch mobilenet_v2 --epochs 1 --prune 0.3 --finetune-epochs 1 --no-pretrained --workers 0`
  - 結果：263 train / 57 val / 61 test，跳過 17 張壞檔，有印出警告。
  - base 和 pruned30 都有印出 test loss，`.json` 裡有 `split`、`n_test`、`test_loss`。✅
- 指定 `--val-ratio 0.2` 時會印出被忽略的提示。✅
- 舊的單一資料夾：照常隨機切，`n_test` 是 0。✅
- 註解規則檢查（data_loader、train、live_demo）：0 個問題。✅
- sandbox 的共享記憶體太小，batch 32 加上多個 worker 會出現 `No space left on device (shm)`；這是 sandbox 的限制，PC 不受影響，改用 `--workers 0` 測。

## 第十四階段（移植隊友部署包的功能、重構訓練、新版 live_demo、TRT 轉換腳本，2026-11-03）

使用者提供隊友的部署包 `jetbot_bundle_spatial_single.zip`（只有部署用程式，沒有訓練程式；壓縮檔裡少了權重檔 `best_steering_model_xy.pth`）。比較後使用者決定全部移植。使用者的要求和決定：
- 每個網路的訓練流程寫在各自的檔案裡，成為一個函式；`train.py` 依任務呼叫不同網路。
- 多 seed 訓練做成 parser 選項。
- 控制選項一律是按鈕或滑桿。控制公式先用隊友的。
- `live_demo`：「允許移動」改成單一按鈕（按一下開始、再按一下停止，顏色和文字跟著變）；載入模型就開始推論，停車不停推論；移除「開始推論」按鈕；沒有車也要能看介面；沒有鏡頭時顯示「未連接鏡頭」；一開始顯示「未載入模型」，等使用者選。
- **下拉選單只列 TRT 模型**。新增車上轉換用的 `.py`；上傳的模型和轉換後的模型放在兩個資料夾（`models/`、`models_trt/`），都進 gitignore。
- CoordConv 沒有移植（第一題使用者沒有回答，之後要的話再加）。

改動：
- `resnet18.py`、`mobilenet_v2.py`：新增 `DEFAULTS`（epochs 70、lr 1e-3、finetune 15 / 1e-4）和 `train()`（Adam + MSE，依 validation MSE 保留最好的 epoch，記錄 val 像素誤差）。
- `metrics.py`（新）：`evaluate()` 回傳 mse、mae_x_px、mae_y_px、mean_px、p95_px（1 單位 = 112 px）。`mse` 和舊版 `evaluate` 的數值相同。
- `sanity.py`（新）：`file_sha256`、`make_sanity`（8 張 validation 畫面，標完車道線後存 PNG base64，加上 CPU FP32 的輸出）、`check`（FP32 容許 1 px，FP16／TRT 容許 2 px；結果是 passed／failed／missing）。
- `train.py`：
  - 改成呼叫 `arch.train()`；`--epochs` 等參數不給時用網路的 `DEFAULTS`。
  - `--seeds`（多 seed，檔名加 `_s<seed>`，輸出 `<name>_seeds.json` 平均 ± 標準差）。`--split-seed` 取代 `--seed`，切分保持固定。
  - `.json` 新增 `val_metrics`、`test_metrics`、`weights_sha256`、`sanity`（約 0.9 MB）。`best_val_loss`、`test_loss` 保留（MSE）。
  - `--base` 不能和多個 seed 一起用。
- `loader.load_model`：`.json` 有 `weights_sha256` 時先比對。`compression.wrap_trt_engine` 會寫入新檔的 SHA256。
- `control.py`：改寫成 `SteeringController.update(xy, now) -> (left, right, status)`。隊友公式（預設）／投影片公式、角度低通、PID（D 為每秒、I 限制 ±0.3）、速度介於 min／max、轉彎減速、slew、馬達上限 0.4、內側輪反轉；NaN／超出範圍／目標太近時回傳非 ok。`PDController` 和 `model_output_to_xy` 已移除。
- `convert_trt.py`（新，車上用）：預設 torch2trt（FP16，從 `.pth` 重建）；`--method trtexec`（需要 `.onnx`）。先檢查來源 SHA256 → 轉換 → 寫 `.json`（label 加「| TensorRT」、新 SHA256）→ 從硬碟重新載入 → sanity 比對 → 量延遲 → 寫入 `trt_validation`。已轉過的會跳過（`--force` 重轉）；一個失敗不影響其他。
- `live_demo.ipynb`：重寫，功能見上方「live_demo.ipynb 功能」。
- `camera.draw_overlay`：預測是 NaN／inf 時不畫（見下方「已測」）。
- `.gitignore` 加上 `/models_trt/`。README 全面更新。

**更正**：上一輪我跟使用者說投影片公式在目標點偏低時「目標在右卻往左轉」，這是錯的。實際上分母變負時，角度會超過 ±90°（轉太多），只有 x 接近 0 時才會在 ±180° 之間跳。`control.py` 的 docstring 和 README 已經寫成正確的說法。

**控制公式會不會影響訓練**：不會。模型學的是畫面到目標點，和控制無關，換公式不用重新訓練。

已測（sandbox，CPU）：
- `train.py --data splits --arch mobilenet_v2 --epochs 1 --prune 0.3 --finetune-epochs 1 --seeds 0 1 --onnx`：兩個 seed 都完成，印出 val／test 的平均 ± 標準差，`_seeds.json` 有寫出。seed 0 的 val MSE 和上一輪單 seed 相同（0.14252），可以重現。✅
- sanity：同一個模型差 0.000 px（通過）；換成另一個 seed 的權重差 24.4 px（失敗）；`.pth` 換掉但 `.json` 沒換時，SHA256 拒絕載入。✅
- 控制：目標在右邊時左輪比較快、在左邊時右輪比較快；NaN／超出範圍／目標太近時回傳 0 和原因；起步有 slew 漸增；max speed 0 → 不動。✅
- `convert_trt.py` 用**假的 torch2trt**（輸出加 1e-3 雜訊模擬 FP16）測流程：兩個模型 PASSED（約 0.2 px）、壞檔回報 SHA256 錯誤且不影響其他、第二次執行會跳過、`list_models` 顯示「| TensorRT」。✅
- `live_demo` 用 nbclient 跑兩種情境（假 TRT 模型）：
  - 沒有鏡頭：顯示「未連接鏡頭」佔位圖；載入模型、比對通過；按開始行駛顯示「未連接鏡頭，無法開車」。✅
  - 輪播資料夾：未選模型時無法開車 → 載入、比對通過 → 開始行駛（按鈕變紅色「停止」、馬達有值）→ 停止（變回綠色、馬達 0、推論繼續）→ NaN 輸出自動停車 → 推論卡 1.2 秒時看門狗停車 → sanity 失敗時無法開車 → benchmark 暫停並恢復迴圈 → 結束。✅
  - **測試中找到並修正一個真的 bug**：模型輸出 NaN 時，`camera.draw_overlay` 把 NaN 轉整數出錯，推論迴圈執行緒就這樣死掉，而且沒有任何提示（車子已經先停了，但之後畫面不再更新、也不能再開車）。舊版 `live_demo` 也有同樣的問題。修正：
    - `draw_overlay` 遇到 NaN／inf 就不畫點和轉向條。
    - 迴圈每一輪都包 `try/except`：出錯時停車、在模型資訊列顯示 traceback，迴圈繼續跑。
    - `benchmark()` 等迴圈暫停最多 10 秒，等不到就報錯，不會無限等待。
    - 修正後連續跑 3 次完整測試都通過，迴圈沒有再出錯。
- Notebook 用 Python 3.6 語法解析通過；Nano 端的模組通過 vermin 3.6 檢查（只有 `compression` 同名的已知誤判）；註解規則檢查 0 個問題。
- **還沒測**：真的 torch2trt／TensorRT 轉換與載入、Nano 上的舊版 ipywidgets 介面（按鈕圖示、ToggleButtons）、實車控制效果。

## 第十五階段（CoordConv：在模型裡加座標通道能不能轉 TensorRT，2026-11-03）

使用者決定：CoordConv 做成**獨立的模型**（不是訓練選項），座標通道在**模型裡面**處理，前處理和讀相機都不改；實際執行的都要是 TRT。隊友的模型用 ResNet 當基底，使用者稍後會提供原始碼，**在那之前不要猜它的結構、不要動手實作模型**。

torch2trt 原始碼分析（github NVIDIA-AI-IOT/torch2trt，看歷史版本）：
- `torch.cat` 的輸入有常數（buffer）時：2019-09-30 起用 `trt_()` 自動建立常數層，2020-09-01 起改用 `add_missing_trt_tensors`。這兩個時間點之後的版本**都能轉**。更早的版本不行。
- `Tensor.expand`（2021-01 才加入）直接讀 `input._trt`，對常數 buffer 做 expand **轉不了**。
- 結論：模型在 batch 1（推論和 TRT 追蹤時）直接 `cat([x, coords])`；訓練時 batch > 1 才用 expand（只在 PyTorch 跑）。
- ONNX（opset 11，batch 1）匯出後只有 Concat + 常數 initializer，沒有 Expand；onnxruntime 和 PyTorch 差 < 1e-6。trtexec 應該可以轉。

新增 `check_trt_coord.py`（車上執行，不用馬達和相機）：
- 印出 python／torch／TensorRT／torch2trt 版本，並檢查車上 torch2trt 的 cat 轉換器能不能處理常數。
- 測三種寫法：A `cat_const`、B `cat_expand`（預期失敗）、C `resnet18_coord`（真正模型會用的寫法），每種都用 torch2trt FP16 和 ONNX→trtexec 轉換，和 PyTorch FP32 比較（容許 2 px）。
- sandbox 只能跑 PyTorch／ONNX 的部分，已確認可以執行；**TensorRT 部分要使用者在車上跑**。

## 第十六階段（移植隊友的 CoordConv / DSNT，並在沒有 GPU 的情況下驗證 TRT 轉換，2026-11-03）

來源：使用者上傳的 `model_training_methods.zip`（隊友的訓練原始碼）。比較結果已回覆使用者：
- zip 裡有三套流程：① `train.py`（RGB、AdamW、部分凍結、BN 固定、cosine、Huber、用 px 選模型；它的 `experiments.json` 沒附）② `course_comparison.py`（課堂配方）③ `spatial_comparison.py`（課堂配方 + CoordConv / DSNT / 兩者，seed 17/29/43）。
- 資料切分和我們拿到的 `splits` 逐檔相同（277 / 60）。
- ①的訓練技巧**還沒移植**，等使用者決定。

新增三個網路（都用 `--arch` 選，每個網路一個檔）：
- `resnet18_coord.py`：座標通道在模型裡加（buffer `coords`，(2,H,W)），`conv1` 5 通道，多出的權重從 0 開始。batch 1 用 `view` 接上座標圖，訓練 batch > 1 才 `expand`。
- `resnet18_dsnt.py`：layer4 stride 1、1×1 卷積熱圖頭、28×28 softmax、期望座標；損失 MSE + 0.1·JS（`resnet18.train` 新增 `loss_fn` 參數）。bilinear 放大和期望值改成**固定不訓練的 Linear（bias 為 0）**，避開 `interpolate` 和 `torch.softmax`。
- `resnet18_coord_dsnt.py`：兩者合併。
- `loader.detect_arch` 改用各網路檔的 `KEYS`（全部存在、key 最多者勝出），取代 `HEAD_KEY`。
- `compression.prune_model`：DSNT 熱圖頭最後一層（1 通道）不剪。

PyTorch 端已測（`scratchpad/newmodels/test_models.py`）：
- 座標圖 = 隊友 meshgrid（差 6e-8）；預訓練剛載入時 coord 輸出 = 普通 ResNet-18（差 0）。
- DSNT 和隊友原寫法（interpolate + flatten softmax）差 < 3e-7，熱圖也一樣。
- batch 4 和 batch 1 結果一致；5 種網路都能正確判斷；訓練後固定層和座標圖不變。
- 剪枝 30% → 存檔 → 依形狀重建：差 0。MACs：coord 1.90G→0.89G、dsnt 3.06G→1.43G、coord_dsnt 3.14G→1.48G。
- ONNX（opset 11）+ onnxruntime 差 < 3e-7；op 只有 Add/Concat/Constant/Conv/Flatten/Gemm/GlobalAveragePool/Identity/MaxPool/Relu/Reshape/Softmax。

**離線 TRT 驗證**（使用者要求「一定有辦法，去查」）：新增 `check_trt_offline.py` + `fake_tensorrt/`。
- 做法：用**真的 torch2trt 原始碼**（git checkout 指定版本），接上模擬的 TensorRT API；torch2trt 照常建網路，模擬器依 TensorRT 語意（implicit / explicit batch）用 PyTorch 逐層算出結果。
- 用輸入 A 轉換、輸入 B 比對，所以被偷偷凍成常數的值也抓得到；torch2trt 印出 unsupported method 也判 FAIL；輸出幾乎不隨輸入變化時標 WEAK（測試權重太弱）。
- 抓不到：Nano 上建 engine 的限制、FP16 誤差 → 車上 `check_trt_coord.py` 仍是最後確認。
- 執行 GitHub 下載的 torch2trt 原始碼，使用者已同意（2026-11-03）。
- 結果（torch2trt 2019-10、2020-05、2020-09、v0.2.0、v0.3.0、v0.4.0 × TensorRT 7.1.3 / 8.0.1）：

  | 模型 | 結果 |
  |---|---|
  | resnet18、mobilenet_v2、resnet18_coord、resnet18_dsnt、resnet18_coord_dsnt | 全部 OK（誤差 ≤ 0.001 px） |
  | 對照：直接 cat (1,2,H,W) 常數 | 2020-09 前 FAIL（常數保留 batch 維度，已從原始碼確認），之後 OK |
  | 對照：expand 常數 | 全部 FAIL |
  | 對照：`torch.softmax` | 全部 FAIL（被凍成常數，差 2.8 px） |
  | 隊友原本的 DSNT | 2019 版被凍成常數（差 56 px），之後用到 resize（模擬器沒做；TensorRT 7 的 resize 沒有 half-pixel 模式） |

  第一版模型（buffer (1,2,H,W)、Linear 無 bias）在 2019 版失敗，所以改成 `view` 和 bias=0 的寫法。torch2trt v0.5（2024，動態形狀）用到模擬器沒做的層，不在範圍內（JetPack 4 也不用它）。
- `check_trt_coord.py`（車上）改成直接測 pipeline 的 4 個 ResNet 網路 + 2 個對照，同樣用兩組輸入比對。

## 第十七階段（訓練實驗 notebook，2026-11-03）

使用者要求：寫一份 ipynb 跑訓練，每格一個訓練指令（`epochs = ...`、`model = ...`），印 test 結果；Run All 時先檢查同名檔案，已訓練過就不重訓；安裝環境的指令也要寫進去。使用者選擇：檔名自動由設定產生、跳過時印上次的 test 結果、要總表、實驗放 5 種網路＋augmentation 比較、不剪枝。

- 新增 `project4/train_experiments.ipynb`（17 格）：第 0 節安裝（沒有 pip 時先 `ensurepip`）、共用設定與 `run()`、5 種網路、ResNet-18 × 4 種 augmentation、總表。
- `run()` 用 subprocess 呼叫 `train.py --name <自動檔名>`，即時印出訓練 log；失敗時丟例外，讓 Run All 停在那一格。
- 單一 seed 檢查 `<name>_base.pth` + `.json`；多 seed 檢查 `<name>_seeds.json`，印平均 ± 標準差。
- `requirements-pc.txt`：`opencv-python` → `opencv-python-headless`（sandbox 測試時一般版缺 libGL 無法 import）。
- 已測（sandbox，1 epoch、真實 splits）：第一次 Run All 每個實驗都有訓練，與前面重複的 resnet18+general 自動跳過；第二次 Run All 全部跳過並印出相同結果；2 個 seed 的路徑和總表正確。

## 第十八階段（augmentation 改成 powerset、移除車上輸入干擾，2026-11-04）

使用者要求：augmentation 只剩一個開關（開 / 關），各方法沒有自己的開關；開的時候用 7 種方法（翻轉、旋轉、裁切、顏色、噪聲、模糊、壓縮）的 powerset 擴張訓練集；新增隨機方向的動態模糊；batch size 預設 16、epochs 10；車上推論**不應該主動加噪聲**。使用者選擇：組合內參數每次隨機、保留空集合（原圖）、模糊 3~15 px、方向 0~360°。

背景：使用者第十七階段的訓練結果（1660 Ti，單一 seed）是 resnet18 none 20.5 px 反而比 gen 26.7 px 好；舊版 `general` 每張圖都會做顏色抖動，所以訓練時完全看不到原圖。

- `augmentation.py`：`Augmenter(enabled)`，`aug(image, x, y, subset)`；`METHODS`、`SUBSETS`（128 種，空集合在 index 0）、`n_variants`、`tag()` = `aug` / `none`。新增 `motion_kernel()`、`motion_blur()`（cv2.filter2D，邊緣鏡像）。方法順序固定：flip → rotate → crop → color → noise → blur → compress。拿掉所有「機率觸發」（perlin_p、compression_p、hflip_p）；壓縮內部仍是 JPEG / 144p 各一半。
- `data_loader.py`：開 augmentation 時 `len = 圖片數 × 128`，樣本 `i` = 第 `i % 圖片數` 張圖 + `SUBSETS[i // 圖片數]`；`make_loaders(directory, augment, ..., batch_size=16)`；`split_info.train` 記的是圖片數。
- `train.py`：`--augment` 取代 `--general/--perlin/--compression`；`--batch-size` 預設 16；`.json` 的 `augmentation` = `{enabled, method: powerset, methods, variants_per_image, tag}`，另記 `n_train_samples`；`--base` 讀舊 `.json` 時任一舊開關為真就當作開。
- `resnet18.py`、`mobilenet_v2.py`：`DEFAULTS["epochs"]` 70 → 10（其他三個網路沿用 resnet18）。finetune 仍是 15 epochs，沒有改。
- `camera.py` 刪掉 `degrade()`；`live_demo.ipynb` 刪掉兩個「輸入干擾」按鈕、迴圈裡的 `degrade` 呼叫、CSV 的 `perlin` / `compression` 欄。車上畫面現在是 `camera.read()` → `mark_lanes` → 推論，中間沒有其他改動。
- `train_experiments.ipynb` 重新產生（18 格）：`run(arch, epochs, lr, augment, batch_size=16, seeds)`；第 1 節 5 種網路 augmentation 開、第 2 節 5 種網路 augmentation 關。舊的輸出（第十七階段的結果）不在新 notebook 裡；`models/` 裡的舊檔名（`_gen_e70_..._bs8`）不會和新檔名衝突。
- `preprocessing.py` 說明文字、README 同步更新（README 也回答了「這樣還算 augmentation 嗎」：算，offline augmentation）。
- 已測（sandbox CPU，每個 split 3 張）：`len` = 384、128 種組合都產生 224×224 且標籤在畫面內、空集合回傳原圖；`train.py --arch resnet18_dsnt --augment --epochs 1 --prune 0.3` 完整跑完；`--base` 沿用 `aug` 設定；7 種方法各自和全部疊加的預覽圖目視正確（標籤跟著翻轉 / 旋轉 / 裁切移動）。
- 訓練量：一個 epoch 277 × 128 = 35,456 張；10 epochs ≈ 22k 步（batch 16），約為舊設定（70 epochs × 277 / 8 ≈ 2.4k 步）的 9 倍。

## 第十九階段（augmentation 改成最多疊 2 種方法，2026-11-04）

使用者回報開了 augmentation 表現不好，懷疑旋轉 / 裁切後座標沒跟著動。

- 驗證標籤跟隨（在標籤位置畫高斯亮點，變換後量亮點中心，各 300 次）：翻轉 / 旋轉平均 0.02 px；裁切平均 0.05 px，最大 2.4 px（亮點在裁切框邊緣被切掉一半，量測誤差，不是 bug）；三者疊加平均 0.06 px。座標有正確跟著動。
- 推測主因是完整 powerset 太重（平均每張疊 3.5 種、7/8 的圖含模糊 / 壓縮 / 噪聲）；一般做法（AutoAugment 2 種、RandAugment 1~3 種）一張只疊少數幾種。使用者同意改成最多 2 種。
- `augmentation.py`：`MAX_METHODS = 2`，`SUBSETS` 只留大小 ≤ 2 的組合：原圖 1 + 單一 7 + 兩兩 21 = 29 種；`tag()` 改成 `aug2`（和舊的 128 種 `aug` 檔名分開，notebook 不會誤跳過）。`.json` 多記 `max_methods`。
- `data_loader.py`、`train.py` 說明、README、notebook 文字同步（×128 → ×29）。notebook 重新產生。
- 一個 epoch 277 × 29 = 8,033 張；預設 10 epochs、batch 16 ≈ 5k 步，約為舊設定（2.4k 步）的 2 倍。
- 已測：29 種組合都能跑、tiny split 長度 3 × 29 = 87。
- `train_experiments.ipynb` 的實驗依使用者要求換成 5 種網路 × augmentation 開 × 10 epochs（檔名 `<網路>_aug2_e10_lr0.001_bs16_seed0`），拿掉 augmentation 關的那 5 格。

## 第二十階段（移植隊友流程 ① 的訓練方式，2026-11-10）

使用者對結果不滿意，討論後決定先移植隊友流程 ① 的訓練方式（網路維持簡單）。使用者選擇：直接取代課程那套、超參數用常見值（隊友的 experiments.json 不在 zip 裡）、loss 預設 MSE 可選 Huber、MobileNetV2 比照 ResNet 部分凍結。

- 背景：使用者用 70 epochs 跑的 aug2 結果（`a99f7be`）：coord_dsnt 21.0、dsnt 22.5、resnet18 22.9、coord 24.4、mobilenet 24.8 px；x 誤差 9~12 px、y 16~21 px（y 約是 x 的兩倍）。
- 新增 `training.py`：`fit()` 共用迴圈。部分凍結（`TRAINABLE` 前綴；原本就凍結的參數不會被解凍，結束時恢復原本的 requires_grad）、BN 全部 eval、AdamW 兩組 lr（head = lr、backbone = lr × 0.1）、weight decay 1e-4、cosine 到 1e-6、clip 5、MSE 或 Huber（beta 0.1）、用 validation `mean_px` 挑模型、`min_epochs` 10 後 `patience` 5 提早停止、非有限 loss 直接報錯。回傳的 metrics 多 `best_epoch`，history 多 `lr`。
- `resnet18.py`、`mobilenet_v2.py`：`train()` 改呼叫 `training.fit`；`DEFAULTS` 加 `backbone_lr_scale`、`weight_decay`、`patience`、`min_epochs`；新增 `TRAINABLE` / `HEAD`。MobileNetV2 訓練 `features.0`、`features.11~18`、`classifier`。
- `resnet18_dsnt.py`：`TRAINABLE = backbone.conv1/layer3/layer4 + head`、`HEAD = head`；`make_loss(..., coordinate)` 座標項可選 MSE / Huber。`resnet18_coord.py`、`resnet18_coord_dsnt.py` 轉傳 `loss`。
- `train.py`：`--loss mse|huber`；`.json` 多 `training`（recipe 與數值）。剪枝後 finetune 也用同一套（lr = finetune_lr）。
- `train_experiments.ipynb`：每格多 `loss = "mse"`，檔名多 loss 一段（`resnet18_aug2_mse_e10_...`），不會和舊訓練方式的檔案撞名。
- 已測（sandbox CPU，tiny split）：5 個網路各跑 2 epochs，被改到的參數全部在 `TRAINABLE` 內、BN running stats 沒變、DSNT 固定層仍 requires_grad=False、結束後 flag 恢復；提早停止（patience 2、min 3）在第 4 epoch 停；`train.py --augment --loss huber --prune 0.3` 完整跑完；notebook `run()` 第二次正確跳過。
- 注意：預設 epochs 10 = `min_epochs`，所以 10 epochs 時提早停止不會觸發；要讓它作用，epochs 要設大一點（例如 30~70）。

## 未完成 / 未驗證

- 隊友流程 ① 的訓練方式已移植（第二十階段），但還沒在 GPU 上用真實資料訓練、和舊結果比較。
- CoordConv / DSNT 還沒有用真實資料完整訓練過，也還沒在車上轉 TRT（離線模擬全部 OK）。
- **TensorRT 路徑完全沒有測過**（sandbox 沒有 GPU／TensorRT）：`load_model` 讀 TRT `.pth` 的部分、`_trt_input_dtype`、`TRTModule` 跟 trtexec 建出的 explicit-batch engine 是否相容（前次交接列的最大風險）。
- **Nano 實測進度**：使用者在板子上跑過 live_demo，修好舊版 torchvision 的 `TypeError` 後，模型能載入、畫面能更新（第三階段）。以下**還沒在板子上驗證**：
  - 換模型時的 flush（jetbot `Camera.stop()` / `start()` 重啟要多久、穩不穩）、`'csi'` 來源的 GStreamer 管線字串。
  - 髮夾彎擴充（轉彎減速、內側輪反轉）的實際行駛效果。
  - 前處理在板子上的速度（每幀都做）與效果。
- 第十七階段的正式訓練結果已回報（見第十八階段背景），但都是單一 seed、舊版 augmentation；使用者用 128 種組合的版本訓練過 5 epochs（`_aug_e5_`），表現不好；改成 29 種（`aug2`）後還沒在 GPU 上訓練過。
- 剪枝後的 finetune 仍是 15 epochs；augmentation 開時每個 epoch ×128，要不要一起降低還沒決定。
- 還沒做的訓練改進：合成偏離畫面（augmentation）、Huber loss、推論平滑。
- 測試腳本（假 jetbot、nbclient 模擬）和 notebook 產生器 `make_nb.py` 都沒有進 repo，使用者決定不保留（會在車上實測）。從第十一階段起 `live_demo.ipynb` 直接修改，不再用產生器重新產生。

## 測試指令與結果

以下是第一階段的測試；之後各階段的測試寫在該階段的「已測」裡。

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

1. 使用者在本機開 `live_demo.ipynb` 看介面有沒有問題（PC 沒有 TRT，下拉選單會是空的；要看推論畫面可以暫時把 `MODELS_DIR` 改成 `models`，`CAMERA_SOURCE` 改成 `'opencv'` 或 `'folder'`）。
2. 補齊那 17 張壞檔（請隊友重傳），解壓到 `project4/data/splits/`；PC 上用 `--data data/splits` 訓練，用 test 的像素誤差（可加 `--seeds 0 1 2`）比較各種 augmentation 和模型。
3. 板子上跑 `python3 convert_trt.py`，驗證 torch2trt 轉換和 sanity 比對（風險最高）；失敗就試 `--method trtexec`。驗證換模型的 flush。
4. 在跑道上調控制參數（隊友的起始值還沒調過）；比較隊友公式和投影片公式。
5. 板子上比較 ResNet-18 和 MobileNetV2 的延遲（`convert_trt.py` 的 `trt_validation` 和 `benchmark()`）。
6. 車上跑 `python3 check_trt_coord.py`，確認 4 個 ResNet 網路在真的 TensorRT 上 OK（離線模擬已全部 OK）；結果貼回來。
7. 用新的 `train_experiments.ipynb`（5 種網路 × augmentation 開 / 關）正式訓練，比 test 像素誤差（建議 `--seeds 17 29 43`，和隊友一樣）。DSNT 運算量約 1.7 倍，要看 Nano 的延遲。
8. 用新訓練方式重跑 `train_experiments.ipynb`，和第二十階段背景的舊結果比較；之後再決定要不要把轉向角誤差加進評估。pure pursuit：使用者決定**不做**（2026-11-03）。
