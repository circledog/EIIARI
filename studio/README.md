# studio：JetBot 網頁工作室

在車上（或 PC 上）開一個網頁，像 Jupyter 一樣用瀏覽器連進來：輸入密碼登入 → 選任務 → 調參數 → 啟動 → 看即時畫面。

- **只在區網使用**（沒有 HTTPS）。不要把 port 開到外網。
- 網頁行程（tornado）不載入 torch；每個任務在自己的「任務行程」（runner）裡跑，只有它碰相機和馬達。網路再塞也只影響網頁行程。

```
瀏覽器 ──HTTP / WebSocket──► 網頁行程（tornado，python3 -m studio）
                               │ 127.0.0.1 TCP（JSON 標頭 + JPEG）
                               ▼
                           任務行程（python3 -m studio.runner）：相機、推論、控制、馬達
```

## 安裝與啟動

### PC（訓練用、或沒有車時看介面）

```bash
pip install tornado            # 其他套件同 project4/requirements-pc.txt
cd EIIARI                      # repo 根目錄（有 studio/ 的那層）
python3 -m studio set-password # 第一次：設定登入密碼（至少 6 個字）
python3 -m studio --camera dummy   # 沒有相機時用假畫面；有 webcam 可用 --camera opencv
```

瀏覽器開 `http://<電腦 IP>:8890`。

### 車上（Jetson Nano，在 jetbot 的 Jupyter 容器裡跑）

車上的 torch、jetbot、相機函式庫都在 Jupyter 的 docker 容器裡，所以 studio 也要在容器裡跑。Jupyter 本身就依賴 tornado，**不用多裝套件**。

```bash
# 在板子上（host）：第一次設定密碼（存在容器的 /workspace/.studio，重開機不會消失）
docker exec -it -e STUDIO_HOME=/workspace/.studio -w /workspace/EIIARI jetbot_jupyter python3 -m studio set-password

# 手動啟動（測試用）
docker exec -it -e STUDIO_HOME=/workspace/.studio -w /workspace/EIIARI jetbot_jupyter python3 -m studio

# 開機自動啟動（systemd）
sudo bash studio/deploy/install.sh
```

- 容器名稱、repo 路徑不同的話，改 `studio/deploy/studio.service` 開頭的 `CONTAINER`、`REPO`。
- jetbot 的容器通常用 `--network host`，所以 8890 可以直接從區網連。連不到的話，看容器啟動時有沒有對應的 port。
- **相機同時只能被一個程式使用**：開著 `live_demo.ipynb` 時，studio 的任務會打不開相機（任務頁會顯示原因）。反過來也一樣。

選項：`--port`、`--address`、`--role car|pc`（預設自動判斷：有 `/etc/nv_tegra_release` 就是車）、`--camera jetbot|csi|opencv|folder|dummy`、`--camera-folder`、`--debug`。

## 頁面

- **首頁**：任務清單（開發中的任務灰色顯示）、目前執行中的任務、背景工作（訓練、TRT 轉換）、硬碟配額。上方一直顯示 CPU／GPU／RAM／硬碟，資源不足時出現黃色警告。
- **任務頁**：左邊即時畫面＋啟動／停止，右邊參數表單＋preset（載入／另存／刪除），下面即時狀態和事件紀錄。
- **緊急停止**：每一頁右上角都有紅色按鈕，快捷鍵是**空白鍵**（在輸入框裡打字時不算）。

## 安全機制

| 狀況 | 結果 |
|---|---|
| 按緊急停止／空白鍵 | 馬上停馬達（測試：0.05 秒） |
| 所有開著的任務頁都沒有回應超過「斷線容忍時間」（關掉分頁、Wi-Fi 斷線、網頁行程卡住） | 停馬達。預設 **5 秒**（使用者決定：短暫的 Wi-Fi 不穩不要讓循線的車停下來）；手動駕駛時可以調短，最短 0.3 秒。測試：設 0.3 秒時 0.55 秒停車 |
| 沒有任何頁面在回應時要開動 | 拒絕（`link_down`） |
| 網頁行程當掉／被 kill | 任務行程的連線斷掉 → 馬上停馬達並結束（測試：0.15~0.65 秒） |
| 任務行程當掉／被 kill | 網頁行程在 0.25 秒內發現，車上會執行 `python3 -m studio.motor_guard` 直接透過 I2C 停馬達（**還沒在實機測過**） |
| 任務行程收到 SIGTERM／Ctrl-C | 先停馬達再結束 |

頁面每 0.1 秒送一次心跳；網頁行程只在最近 0.3 秒內有頁面心跳時，才轉送心跳給任務行程；只有任務行程套用「斷線容忍時間」（兩邊都套的話時間會加倍，測試時發現並修正）。

## 影像串流（不排隊）

- 任務行程只保留最新一筆畫面（`LatestSlot`）。
- 網頁行程只在**有瀏覽器準備好**時才向任務行程要下一張；瀏覽器**顯示完**這張並回 `ack` 之後，才會收到下一張。網路慢 → FPS 變低，但畫面永遠是最新的，不會越積越多。
- JPEG 品質依來回時間自動調整：超過 0.5 秒降 10（最低 30），低於 0.15 秒升 5（最高 85）。
- 測試：不回 ack 時 1.5 秒內不會收到第二張；回 ack 之後馬上收到最新的一張（中間的都跳過）。

## 資源與配額（Jetson Nano 4 GB）

`studio/config.py` 的 `BUDGET_MB`：

| 項目 | 預算 |
|---|---|
| 模型（`project4/models` + `models_trt`） | 600 MB |
| 資料集與錄影（`~/.studio/recordings` + `project4/runs`） | 2000 MB |
| 地圖（M4） | 300 MB |
| 日誌（伺服器日誌每檔 1 MB × 5 輪替、任務行程日誌、工作日誌） | 100 MB |
| 硬碟保留空間 | 700 MB |
| RAM 警告門檻（可用記憶體） | 500 MB |
| 網頁行程 RAM | 150 MB |
| 任務行程 RAM | 1500 MB |

- 同時只能有一個任務使用相機和馬達；第二個任務會被拒絕並顯示原因。
- 背景工作（訓練、TRT 轉換）一次一個，排隊執行；TRT 轉換需要 GPU 和大部分 RAM，有任務在跑時會等。工作的紀錄在 `~/.studio/jobs/<id>/`，關掉瀏覽器不影響；伺服器重開時正在跑的工作會標成「中斷」。

## 任務：車道線跟隨（`lane_follow`）

把 `project4` 包成網頁，程式都是從 `project4/` 匯入的，沒有複製。

| 頁面 | 在哪裡用 | 內容 |
|---|---|---|
| 任務頁 `/task/lane_follow`（行駛） | 車（電腦上用 PyTorch 模型＋假馬達也能看） | 選模型 → 載入時 sanity 比對 → 即時畫面（綠點、轉向條）→ 開始行駛／停止、錄製 CSV、推論速度比較；右邊是 `live_demo` 的全部控制選項（⚡ 可在行駛中改）；控制週期 p50／p95 和各段耗時 |
| `/lane/datasets` | 電腦 | `project4/data` 底下的資料集（單一資料夾或 train/validation/test）、張數、縮圖（綠點＝標籤）、切換「車道線標示後」、檢查壞檔 |
| `/lane/training` | 電腦（車上會被拒絕） | 表單對應 `train.py` 的每個選項（預設值和 `train.py`／各網路 `DEFAULTS` 相同，有測試檢查）；背景執行、即時 loss 曲線（val px）、進度和剩餘時間、取消；GPU 記憶體不足時自動把 batch 減半重試（最多 3 次） |
| `/lane/models` | 兩邊 | `models/`、`models_trt/` 的模型與 `.json`（val／test 誤差、sanity、TRT 比對結果與延遲）、勾兩個並排比較、下載、上傳、刪除；車上可以「轉 TensorRT」（背景工作，有任務在跑時會等） |

模型從電腦到車上：電腦的模型庫按「下載」（`.pth` 和 `.json`），到車上的模型庫「上傳」，再按「轉 TensorRT」。

行駛頁的模型資料夾預設「自動」：車上用 `models_trt/`（和 `live_demo` 一樣只用 TRT），電腦上用 `models/`。資料夾位置可以在 `config.json` 用 `lane_data_dir`、`lane_models_dir`、`lane_trt_dir` 改。

## 任務：避障＋點選目標前進（`navigate`）

- 在畫面上**點一下**設定目標（換算座標時用圖片實際顯示的大小和位置）；追蹤器（模板比對）每幀找到目標，追丟就停車。
- 空地估計不用模型：起步時正前方那一塊當作地板，學它的顏色；畫面下半部分成 N 個扇區，每個扇區算出「可以往前走多遠」（畫面上的半透明色塊）。白膠帶算地板。
- 規劃：目標所在的扇區夠空就往目標走，不夠就往最近的空扇區繞；**正前方太近就停車**（最後一道保護，不依賴模型）。輸出和車道線模型一樣是一個目標點，所以直接交給 `project4/control.py`。目標夠低（靠近車子）就算到達。
- 標註（互動式模仿學習）：每次點選＝`xy_<x>_<y>_<id>.jpg`（和 dataset_xy 同格式，`train.py` 可以直接用）、緊急停止＝「前方有障礙」、沒人介入的行駛畫面＝「正常行駛」（每秒最多一張）。存在 `~/.studio/recordings/navigate/`，有配額。
- 學習版：下載標註 zip → 電腦上解壓到 `project4/data/` → 訓練頁選 `navigate_labels/xy` → 轉 TRT → 在任務頁載入、模式切成學習版。停車規則仍然有效。
- 為什麼不用深度模型：見 `tasks/navigate/SPIKE.md`（Nano 4 GB 的 RAM／GPU 要留給控制模型；地板比對在 CPU 上每幀幾毫秒）。

## 寫一個新任務

在 `studio/tasks/<任務名>/` 建資料夾：

- `__init__.py`：只放宣告 `TASK = {...}`（不能 import torch，網頁行程會讀它）。欄位見 `studio/tasks/__init__.py`，參數格式見 `studio/params.py`。
- `runtime.py`：`TaskRuntime` 的子類別（`studio/runner.py`），在任務行程裡跑。要做的事：把要顯示的東西放進 `self.slot`、實作 `render()`（轉成 JPEG）、`command()`、`stop_motion()`（**一定要馬上停馬達、不能丟例外**）、`is_moving()`、`telemetry()`。會讓車子動的指令列在 `MOTION_COMMANDS`。
- `page.html`（選用）：放在任務頁畫面下方的自訂內容和 script，可以用 `window.studio.control.command(name, args)`。
- `web.py`（選用）：`HANDLERS = [(pattern, Handler), ...]`，任務自己的頁面或 API（同樣不能 import torch）。

感測器：`ctx.sensors.get("camera")`。里程計、IMU、距離感測器的介面在 `studio/sensors.py`，目前車上只有鏡頭，`get()` 會回傳 `None`；之後加驅動只要繼承 `Sensor`。

介面文字：Python 裡一律英文（ASCII 規則），頁面用 `static/i18n.js` 翻成中文。

## 檔案

| 檔案 | 內容 |
|---|---|
| `__main__.py` | `python3 -m studio`（serve／set-password） |
| `config.py` | 設定檔 `~/.studio/config.json`（密碼雜湊、cookie 金鑰、port、相機）、角色判斷、資源預算 |
| `auth.py` | PBKDF2-SHA256 密碼、登入失敗鎖定、簽章 cookie |
| `params.py` | 參數驗證（float／int／bool／choice／string）、preset |
| `server.py` | 頁面、JSON API、路由 |
| `stream.py` | 影像 WebSocket（ack 背壓）、控制 WebSocket（心跳、緊急停止、指令、即時參數） |
| `runlink.py` | 網頁端：啟動任務行程、轉送心跳、最新一幀、任務行程死掉時跑 motor_guard |
| `runner.py` | 任務行程本體與 `TaskRuntime` 基底類別 |
| `safety.py` | `LinkWatchdog` |
| `jobs.py` | 裝置使用權、背景工作佇列、資源監控、硬碟配額 |
| `ipc.py` | 兩個行程之間的訊息格式 |
| `hardware.py`、`sensors.py` | 馬達（沒有車時用假的）、感測器介面 |
| `motor_guard.py` | 最後手段：直接透過 I2C 停馬達 |
| `tasks/` | 任務：`drive_check`（相機與馬達檢查）、`lane_follow`（車道線跟隨）、`navigate`（避障＋點選目標）、`search`（M4，開發中） |
| `templates/`、`static/` | 頁面與前端（不需要建置工具） |
| `deploy/` | systemd service 與安裝腳本 |
| `tests/` | 測試 |

## 測試

在 `studio/tests/` 裡（PC，不需要硬體；任務行程用假畫面和假馬達，約 40 秒）：

```bash
cd studio/tests && python3 -m unittest discover -s .
```

| 檔案 | 檢查什麼 |
|---|---|
| `test_auth.py` | 雜湊與驗證、鎖定（5 次 60 秒）；沒登入時頁面導向登入頁、API 回 401、WebSocket 被關；錯密碼 403；cookie 被改過無效；POST 沒有 XSRF token 被擋 |
| `test_params.py` | 每種型別的驗證、NaN／inf／布林不算數字、執行中只能改 ⚡ 參數、宣告錯誤會被抓到、preset、透過 API 啟動時範圍外的值被擋 |
| `test_runner.py` | 真的任務行程：串流背壓、沒有頁面時不能開動、斷線停車、緊急停止、網頁行程斷線時任務行程自己停車結束、任務行程被 kill 時跑 motor_guard、第二個任務被拒絕 |
| `test_lane.py` | 車道線任務：兩種資料集格式、表單預設值等於 `train.py` 和各網路的 `DEFAULTS`、**每個表單選項都原樣傳給 `train.py`**（用 `train.build_parser()` 解析檢查）、控制預設值等於 `control.DEFAULTS`、訓練 log 解析與記憶體不足重試、真的用 job 佇列跑一次 `train.py`（1 epoch）、資料集／縮圖／壞檔 API、模型上傳／下載／刪除、電腦上轉 TRT 被拒、真的任務行程：載入模型 → sanity → 開車 → 畫面有預測值 → 即時改參數 → benchmark → 停車 |
| `test_navigate.py` | 空地估計（乾淨地板、正前方箱子、20 張合成圖判斷「前方有障礙」20/20）、追蹤器（轉 30° 仍追到、離開畫面／換場景會追丟）、規劃器規則、輸出可以直接給 `control.py`、標註格式能被 `project4/data_loader` 讀、配額滿時拒絕；真的任務行程：沒目標不能開、點選存標註、開往目標、緊急停止存「前方有障礙」、到達後停車、zip 下載 |
| `test_jobs.py` | 工作佇列（執行、日誌、進度解析、失敗、取消、等 GPU、重啟後標成中斷）、資源與配額、`LinkWatchdog`、網頁行程不會 import torch／cv2／numpy |

也在 Python 3.6.15 + tornado 6.1 的 docker 裡跑過（和車上相同的 Python 版本），全部通過。
