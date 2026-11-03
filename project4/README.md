# Project 4：AI 道路辨識

JetBot 道路跟隨：模型從相機畫面預測「要前往的目標點 (x, y)」，再換算成轉向角，用 PID 控制左右輪。

```
PC / Colab   Data Loader → Data Augmentation → Preprocessing（即時標車道線）→ Model training → Model compression (DepGraph)
                                                                                                   │ .pth / .onnx
Jetson Nano  convert_trt.py（.pth → TensorRT）
             Camera Input →（輸入干擾）→ Preprocessing（同一個函式）→ Model inferencing (TensorRT) → Motor controlling (PID)
```

## 檔案一覽

| 檔案 | 在哪裡跑 | 用途 |
|---|---|---|
| `train_model.ipynb` | — | 課程原始範例，留作參考 |
| `data_loader.py` | PC | 資料集與 DataLoader |
| `preprocessing.py` | PC + Nano | 前處理：把車道線（白膠帶）標成紅色，彎道也適用；固定開啟，即時做、不存檔 |
| `augmentation.py` | PC（Nano 的「輸入干擾」也會用到） | 資料增強，三組開關 |
| `resnet18.py` | PC + Nano | ResNet-18：網路定義＋它自己的訓練函式 `train()` 與預設參數 |
| `mobilenet_v2.py` | PC + Nano | MobileNetV2：網路定義＋它自己的訓練函式（比較用的 other model） |
| `resnet18_coord.py` | PC + Nano | ResNet-18 + CoordConv：模型裡自己加 x / y 座標通道（隊友的 `course_coordconv`） |
| `resnet18_dsnt.py` | PC + Nano | ResNet-18 + DSNT：熱圖 + 期望值輸出，訓練多一項熱圖損失（隊友的 `dsnt`） |
| `resnet18_coord_dsnt.py` | PC + Nano | CoordConv + DSNT（隊友的 `coordconv_dsnt`） |
| `metrics.py` | PC | 評估指標：MSE 和像素誤差（平均、x / y、p95） |
| `sanity.py` | PC + Nano | 權重 SHA256、sanity 比對畫面（訓練時存、車上比對） |
| `loader.py` | PC + Nano | 所有網路共用：建立、剪枝後重建、載入（PyTorch / TensorRT）、`.json` |
| `compression.py` | PC（`wrap_trt_engine` 在 Nano） | DepGraph 剪枝、ONNX 匯出、TRT 包裝 |
| `convert_trt.py` | Nano | 把上傳到 `models/` 的模型轉成 TensorRT，放到 `models_trt/`，並做比對和計時 |
| `check_trt_coord.py` | Nano | 用隨機權重測每個網路能不能正確轉 TRT（torch2trt 和 trtexec 兩條路） |
| `check_trt_offline.py` + `fake_tensorrt/` | PC | 沒有 GPU 時的 TRT 轉換檢查：真的 torch2trt 原始碼 + 模擬的 TensorRT，逐層算出結果比對 |
| `camera.py` | Nano | 相機輸入與前處理 |
| `control.py` | Nano | 轉向角（隊友公式／投影片公式）＋ PID ＋ 差速控制，含平滑和安全檢查 |
| `train.py` | PC | 依任務呼叫各網路的 `train()`：訓練 → 剪枝 → finetune → 匯出，可多 seed |
| `live_demo.ipynb` | Nano（PC 也能開介面） | 推論＋控制的操作介面，只列 TensorRT 模型 |
| `requirements-pc.txt` | PC | PC / Colab 端套件 |

程式碼註解規則：每行有實質內容的程式碼，上一行都有一條英文註解（常見的 library import、單獨的括號、docstring 除外）。修改程式時請維持這個格式。

這幾個資料夾都不進 git：

| 資料夾 | 內容 |
|---|---|
| `data/` | 資料集 |
| `models/` | PC 上 `train.py` 產生、要上傳到車上的 PyTorch 模型（`.pth` + `.json`，可加 `.onnx`） |
| `models_trt/` | 車上 `convert_trt.py` 轉出來的 TensorRT 模型；`live_demo` 只列這裡的模型 |
| `runs/` | 行駛紀錄 CSV |

## 各模組簡介

### `data_loader.py`

- `XYDataset(directory=None, augmenter=None)`：讀取 `xy_<x>_<y>_<uuid>.jpg`，從檔名取出像素座標，再正規化到 [-1, 1]。
  - `directory` 預設是 `data/dataset_xy_2026-10-21/dataset_xy`。
- `make_loaders(directory, general, perlin, compression, ..., split='random')`：回傳 `(train, val, test, augmenter)`。
  - **資料夾已經分好時**（`directory` 底下有 `train/`、`validation/`，`test/` 可有可無）：直接照資料夾用，`split` 和 `val_ratio` 不起作用。`validation` 用來挑最好的 epoch；`test` 訓練中完全不看，只在每個模型存檔前評一次。沒有 `test/` 時 test 是 `None`。
  - 只有一個資料夾時照下面兩種方式切，沒有 test：
    - `split='random'`：用固定 seed 隨機切 9:1。
    - `split='session'`：讀資料集旁邊的 `metadata/`（每張圖一個同名 JSON，含 `session_id`、`captured_at`），**照拍攝段落切**。同一個 `session_id`（一次收集）裡，相隔超過 `segment_gap`（預設 60 秒）就算新的一段；驗證集一次拿整段，相鄰的相似照片就不會分別落進 train 和 val。沒有 metadata 的圖各自算一段；只有一段時會報錯並提示改用 random。
  - **只有訓練集做 augmentation**，驗證集和測試集一律用原圖（都會標車道線）。
  - 讀不了的圖（空檔、檔案不完整）會跳過，並印出 `warning: skipping unreadable image ...`；跳過的張數記在模型 `.json` 的 `split.skipped_unreadable`。
- 前處理和課程一致：resize 224 → tensor → RGB 轉 BGR → ImageNet 正規化。這樣和車上 `camera.preprocess` 的結果完全相同（已驗證差 0）。
- 修掉課程版的 bug：原本就算 `random_hflips=False` 也會隨機翻轉。

### `preprocessing.py`

**只做一件事：把車道線標出來**，而且**固定開啟、沒有開關**。

- **即時做、不存檔**：訓練時在 DataLoader 裡對每張圖做（`mark_lanes_pil`），車上 `live_demo.ipynb` 對每一幀做（`apply_frame`）。
- **順序在 augmentation 之後**：訓練是「讀圖 → augmentation → 標車道線」，車上是「相機 → 輸入干擾（柏林噪聲／壓縮失真）→ 標車道線」。噪聲對偵測的影響兩邊相同，DEMO 切換「輸入干擾」時，模型遇到的情況和訓練時一致。驗證集一樣會標車道線，只是不做 augmentation。

- **做法**：把看起來像白膠帶的像素塗成純紅色 (BGR 0,0,255)。模型仍然看得到原本的畫面（3 通道，網路不用改），只是車道線被明確標出來。標籤不變。
- **能處理彎道**：逐像素判斷，不是用直線去套。彎道、髮夾彎、鏡頭前面很寬的膠帶都標得出來。
  1. 地板亮度：取畫面下半部中間 LAB 亮度的第 40 百分位。
  2. 膠帶：比地板亮 `delta`（70）以上；或是細的亮條紋（white top-hat 21 px 大於 35，而且比地板亮 40 以上），用來抓遠處比較細、比較暗的膠帶。
  3. 去掉牆壁、場外木地板這類「又厚又在上面」的亮區（厚度超過 31 px，而且位在畫面上方 55% 或碰到上緣）。位置低的厚亮區是鏡頭前的膠帶，所以保留。
  4. 去掉小雜點，以及畫面上方 1/4 的小塊。
- 參數是用 224×224 的畫面調的，其他尺寸會等比例縮放 kernel。參數寫在程式裡，訓練和車上用同一組，也會記進模型 `.json` 備查。
- **在預計訓練的 198 張上測過**：每張都有標出膠帶；沒有任何一張的標籤點（綠點）落在被標成紅色的地方；PC CPU 每張約 6 ms。Nano 上會更慢，`live_demo` 顯示的延遲已經包含前處理時間。
- 已知的不完美：畫面最上方的人、腳、椅子偶爾有小塊被標紅；場外木地板的紋路偶爾有雜點。
- 同一張圖，訓練路徑和車上路徑餵進模型的 tensor 完全相同（已驗證差 0）。
- 想看標出來的結果，用下面的預覽工具。

**預覽工具**（PC）：

```bash
python preprocessing.py review --data <資料夾>                         # 前處理後的圖加上標籤，排成一張總覽
python preprocessing.py review --data <資料夾> --model models/X.pth    # 依模型誤差排序，最差的在前面
```

會產生 `label_review.jpg`：紅色是標出的車道線，綠點是標籤，青色圈是模型的預測。終端機也會依同樣的順序列出檔名。

### `augmentation.py`

`Augmenter(general=False, perlin=False, compression=False)`，呼叫方式是 `aug(image, x, y) -> (image, x, y)`。三個開關**全關時原樣輸出**。

| 開關 | 內容 |
|---|---|
| `general` | 水平翻轉、±10° 旋轉、80~100% 裁切再放大、亮度 / 對比 / 飽和度 ±0.3。幾何變換會同步移動標籤座標 |
| `perlin` | 柏林噪聲做出大片明暗不均，模擬光線和陰影 |
| `compression` | JPEG 低品質（q 5~35）重壓，或像 YouTube 144p 那樣縮小、重壓再放大 |

### 網路檔（`resnet18.py`、`mobilenet_v2.py`、`resnet18_coord.py`、`resnet18_dsnt.py`、`resnet18_coord_dsnt.py`）與 `loader.py`

每個網路一個檔，裡面有**網路定義和它自己的訓練函式**；`train.py` 只負責依任務（baseline、剪枝後 finetune、多 seed）呼叫對應網路的函式。共用的載入功能放在 `loader.py`。

| 檔案 | 網路 | 預設輸出層 | MACs / params（未剪枝） |
|---|---|---|---|
| `resnet18.py` | ImageNet 預訓練 ResNet-18（投影片 P.39、P.45） | `fc = Linear(512, 2)` | 1.82 G / 11.18 M |
| `mobilenet_v2.py` | ImageNet 預訓練 MobileNetV2 | `classifier[1] = Linear(1280, 2)` | 0.32 G / 2.23 M |
| `resnet18_coord.py` | ResNet-18，`conv1` 改 5 通道（RGB + x + y 座標圖） | `fc = Linear(512, 2)` | 1.90 G / 11.19 M |
| `resnet18_dsnt.py` | ResNet-18（layer4 stride 1）+ 熱圖頭 | 28×28 熱圖的期望座標 | 3.06 G |
| `resnet18_coord_dsnt.py` | 上面兩個合起來 | 同 DSNT | 3.14 G |

隊友的三個變體（來自 `model_training_methods.zip`）：
- **CoordConv**：座標通道在**模型裡面**加（隊友是在前處理加），所以相機、前處理、sanity、轉 TRT 的輸入都還是 3 通道，不用改。
  - 座標圖存在權重檔裡（buffer `coords`），PC 和車上一定一樣；數值和隊友的 meshgrid 相同。
  - 多出來兩個通道的權重從 0 開始，所以剛載入 ImageNet 權重時輸出和普通 ResNet-18 完全一樣。
  - batch 1（推論、轉 TRT）用 `view` 接上座標圖；訓練時 batch > 1 才 `expand`。這是離線檢查後選的寫法：`view` 在 torch2trt 2019–2022 每一版都能轉，直接接 (1,2,H,W) 常數在 2020-09 前的版本會失敗，`expand` 在 2021-01 後的版本會失敗。
- **DSNT**：layer4 不降解析度（14×14），1×1 卷積產生熱圖，放大到 28×28、softmax，輸出熱圖的期望座標。訓練損失 = MSE + 0.1 × 熱圖和高斯分布的 JS 散度。
  - 為了能轉 TRT，放大（bilinear）和期望值都改成**固定不訓練的 Linear 層**，數學上和隊友的寫法一樣（實測差 < 3e-7）。避開了舊版 torch2trt 的 `interpolate`（TensorRT 7 的 resize 沒有 half-pixel 模式，值可能不同）和 `torch.softmax`（沒有轉換器，會被偷偷凍成常數）。
  - 運算量約 ResNet-18 的 1.7 倍，Nano 上會比較慢；剪枝 30% 後 1.43 G。
- 選模型、`.json`、剪枝、sanity、live_demo 都跟其他網路一樣，用 `--arch` 選即可。隊友的權重不能直接拿來用（他沒有車道線前處理），要用我們的流程重新訓練。

- 每個網路檔都有：
  - `build(pretrained, num_outputs)`：建立網路。
  - `DEFAULTS`：這個網路的預設 epochs、lr、finetune epochs、finetune lr。`train.py` 沒指定時就用這裡的值。
  - `train(model, train_loader, val_loader, device, epochs, lr, tag)`：這個網路的訓練流程（目前兩個都是 Adam + MSE，保留 validation MSE 最低的 epoch）。要針對某個網路改 optimizer、loss 或學習率排程，只改它自己的檔案，不會影響另一個。
- MobileNetV2 是給報告比較用的 other model：運算量約 ResNet-18 的 1/6，本來就是為行動裝置設計的。剪枝 30% 後是 0.15 G / 1.08 M。
  - 注意：PC CPU 上兩者未剪枝的延遲差不多（depthwise convolution 在 CPU 上效率不好），差距要在 Nano 的 GPU / TensorRT 上才看得出來，請以板子上的 benchmark 為準。
- `loader.py`：
  - `build_model(arch)`：`arch` 是 `loader.ARCHS` 裡的名稱（`resnet18`、`mobilenet_v2`、`resnet18_coord`、`resnet18_dsnt`、`resnet18_coord_dsnt`）。
  - `detect_arch(sd)`：從權重裡的特徵 key（每個網路檔的 `KEYS`）判斷是哪個網路，所以 `--base` 和載入模型都不用另外指定。變體會多幾個 key（例如 `coords`），符合最多 key 的勝出。
  - `rebuild_from_state_dict(sd)`：依權重的形狀重建網路（兩種都適用，depthwise 卷積會保持 depthwise）。剪枝後 channel 數變少，直接 `load_state_dict` 會 size mismatch。
  - `load_model(path)`：自動判斷 `.pth` 是 PyTorch 權重還是 TensorRT 包裝。TRT 的輸入 dtype 從 engine 讀。**`.json` 有記錄 SHA256 時會先比對，對不上就拒絕載入**（檔案沒傳完整，或 `.json` 配錯檔）。
  - `list_models(dir)`：給介面下拉選單用，顯示名稱來自同名的 `.json`。
- 要再加網路：新增一個 `<名稱>.py`（`NAME`、`FILE_TAG`、`LABEL`、`KEYS`、`DEFAULTS`、`build()`、`train()`），再加進 `loader.ARCHS`。加完先跑 `check_trt_offline.py` 確認轉得了 TRT。
- `resnet18.train()` 可以傳 `loss_fn(model, images, labels)` 換掉 MSE（DSNT 就是這樣用），選最佳 epoch 仍然看 validation MSE。

### `metrics.py`、`sanity.py`

- `metrics.evaluate(model, loader)`：回傳 `mse`、`mae_x_px`、`mae_y_px`、`mean_px`（平均距離）、`p95_px`（95% 的圖誤差在這以內）。標籤是 224×224 圖上的座標，1 個正規化單位 = 112 px。像素誤差比 MSE 好理解，例如「目標點平均偏 15 px」。
- `sanity.py`（借用隊友部署包的做法）：
  - 訓練存檔時，在 `.json` 存 8 張 validation 畫面（已標車道線，PNG 無損）和 PC 上 FP32 的輸出，以及 `.pth` 的 SHA256。
  - 車上 `sanity.check(model)` 用同樣的畫面跑一次，最大差超過容許值（FP32 1 px；FP16 / TensorRT 2 px）就算失敗。`convert_trt.py` 轉完會比對一次，`live_demo` 每次載入模型也會比對；**沒通過就禁止開車**。
  - 比對的是標完車道線之後的畫面，所以檢查的是網路和執行環境（剪枝重建、FP16、TensorRT 轉換）。

### `compression.py`

- `prune_model(model, ratio)`：DepGraph 結構化剪枝。
  - 用 torch-pruning 的 `MetaPruner` + `GroupMagnitudeImportance(p=2)`，`fc` 不剪，`round_to=8`。
  - 有殘差連接的層會一起剪，剪完仍是合法的網路（ResNet-18 的殘差、MobileNetV2 的 depthwise 都一樣處理），只是 channel 變少。
- `count_macs_params()`、`export_onnx()`：ONNX 用 opset 11，輸入輸出名稱是 `input_0` / `output_0`。
- `wrap_trt_engine(engine, out)`：把 `trtexec` 建好的 engine 包成 `TRTModule` 能讀的 `.pth`（`convert_trt.py --method trtexec` 會用到）。

| 剪枝比例 | MACs | Params |
|---|---|---|
| 不剪 | 1.82G | 11.18M |
| 30% | 0.84G | 5.27M |
| 50% | 0.49G | 2.80M |

### `convert_trt.py`（在車上執行）

```bash
python3 convert_trt.py                    # models/ 裡還沒轉過的模型全部轉
python3 convert_trt.py models/X.pth       # 只轉這一個
python3 convert_trt.py --force            # 已經轉過的也重轉
python3 convert_trt.py --method trtexec   # 改用 ONNX → trtexec（需要 X.onnx）
```

- 預設用課程的 **torch2trt**：從 `.pth` 重建網路（剪過枝的也可以），內部 FP16、輸入 FP32，不需要 ONNX。
- 每個模型轉完會：從硬碟重新載入（和 `live_demo` 一樣）→ sanity 比對 → 量模型本身的延遲（中位數、p95）。結果寫在 `models_trt/X_trt.json` 的 `trt_validation`。
- 一個模型失敗不影響其他模型，最後會列出每個模型的結果。來源 `.pth` 的 SHA256 不對時，不會浪費時間去轉。

### `camera.py`

- `open_camera(source)`：`'jetbot'`（CSI 相機）、`'csi'`（同一顆相機，但只保留最新畫面）、`'opencv'`（USB / 筆電 webcam）、`'folder'`（輪播資料集圖片）、`'dummy'`（假畫面）。
- `flush()`：丟掉迴圈卡住期間堆積的舊畫面。換模型後會自動呼叫。
  - 原因：jetbot 的 GStreamer `appsink` 不會丟舊畫面。換模型時 TRT engine 反序列化會卡住 GIL，相機讀取執行緒跟著停，畫面就堆起來；之後讀取速度和相機速度一樣，堆積永遠消化不掉，畫面會一直延遲。
  - `'jetbot'` 來源的 flush 是重啟相機管線（約 1~2 秒）；`'csi'` 來源設了 `drop=true max-buffers=1`，本來就不會堆積。
- `preprocess(frame, device, dtype)`：投影片 P.59 的前處理，dtype 依模型決定，不再寫死 `.half()`。
- `degrade()`：在即時畫面上加柏林噪聲 / 壓縮失真，給介面的「輸入干擾」用。
- `draw_overlay()`、`to_jpeg()`：畫預測點和轉向條，供畫面顯示。

### `control.py`

`SteeringController.update(xy, now) -> (left, right, status)`，`xy` 是模型的原始輸出（[-1, 1]，x 左負右正、y 上負下正）。

1. **轉向角**，兩種公式（介面上的按鈕切換，**預設用隊友的**）：

   | 公式 | 算法 | 說明 |
   |---|---|---|
   | 隊友公式 | `v = (y+1)/2`；`angle = atan2(x, max(0.08, 2·(0.95 − v)))` | 分母永遠是正的，目標點在畫面下方也不會出問題；目標點貼到車頭（v ≥ 0.98）就停車 |
   | 投影片公式 | `angle = atan2(x, (0.5 − y)/2)` | 目標點在畫面下方（y > 0.5）時分母變負，角度會超過 ±90°（轉太多）；x 接近 0 時還會在 +180° 和 −180° 之間跳，方向一下左一下右 |

2. **角度平滑**：`angle += α·(raw − angle)`，α = 1 等於不平滑。
3. **PID**：`steering = kp·angle + ki·∫angle + kd·d(angle)/dt + bias`。D 項是「每秒」的變化率，和 FPS 無關；積分限制在 ±0.3，避免累積過頭。
4. **速度**：直線用 max speed，轉彎依「轉彎減速」降低，但不低於 min speed（也不超過 max speed）。
5. **差速**：`left = speed + steering`、`right = speed − steering`，限制在 `[0, 馬達上限]`；開「內側輪可反轉」時下限變成負的。
6. **馬達變化率**：每秒最多變化多少，起步和轉向比較平順；0 = 不限制。

模型輸出 NaN / inf、超出畫面（|x| 或 |y| > 1.25）、目標太近時，回傳的 status 不是 `'ok'`，馬達值為 0，`live_demo` 會停車。預設值來自隊友的 `control.json`，**還沒在跑道上調過**。

**控制公式會影響訓練嗎？不會。** 模型學的是「畫面 → 目標點 (x, y)」，標籤是你們點在畫面上的位置，和控制公式無關。換公式不用重新訓練，同一個模型可以直接比較兩種公式。

### `train.py`

```bash
pip install -r requirements-pc.txt
python train.py --help
```

| 參數 | 說明 |
|---|---|
| `--arch resnet18` / `mobilenet_v2` / `resnet18_coord` / `resnet18_dsnt` / `resnet18_coord_dsnt` | 網路（預設 resnet18）。訓練流程在該網路的檔案裡。前處理（標車道線）固定會做 |
| `--general` `--perlin` `--compression` | augmentation 開關，全不加 = 不做 augmentation |
| `--epochs` `--lr` | baseline 訓練；不給就用該網路檔的 `DEFAULTS`（目前都是 70 / 1e-3） |
| `--batch-size 8` | batch size |
| `--prune 0.3 0.5` | 剪枝比例，可一次給多個 |
| `--finetune-epochs` `--finetune-lr` | 剪枝後的 finetune；不給就用 `DEFAULTS`（目前都是 15 / 1e-4） |
| `--seeds 0 1 2` | **多 seed 訓練**：每個 seed 完整跑一次（baseline + 剪枝），最後印出各模型的平均 ± 標準差，存成 `<名稱>_seeds.json`。只給一個值（預設 0）就是一般訓練 |
| `--split-seed 0` | 隨機切 train / val 用的 seed，多 seed 時保持不變，只有初始化和打亂順序會變 |
| `--onnx` | 同時匯出 ONNX（`convert_trt.py --method trtexec` 才需要） |
| `--base X_base.pth` | 跳過 baseline 訓練，直接剪既有模型（網路從權重判斷，augmentation 設定沿用它的 .json）。不能和多個 seed 一起用 |
| `--split session` `--metadata DIR` `--segment-gap 60` | 照拍攝段落切 train / val（`--metadata` 預設是 `--data` 旁邊的 `metadata/`） |
| `--data` `--out` | 資料集路徑（有 `train/ validation/ test/` 子資料夾就直接照用）、輸出資料夾（預設 `models/`） |

產出：`models/<網路>_<aug>_{base,pruned30,pruned50}.{pth,json}`，例如 `mobilenetv2_gen_pruned30.pth`；多個 seed 時檔名多一段 `_s<seed>`，例如 `mobilenetv2_gen_s1_pruned30.pth`。`.json` 裡有：

- loss 曲線（每個 epoch 的 train loss、val MSE、val 平均像素誤差）
- `val_metrics` / `test_metrics`：MSE、x / y 平均誤差、平均距離、p95（px）
- MACs / params、資料切分、seed
- `weights_sha256` 和 `sanity`（車上比對用，約 0.9 MB，主要是 8 張 PNG）

### `live_demo.ipynb`

| 區塊 | 內容 |
|---|---|
| 模型 | 下拉選單**只列 `models_trt/` 的 TensorRT 模型**，一開始是「未載入模型」，選了才載入。載入完成就開始推論，下方顯示 sanity 比對結果 |
| 連線狀態 | 車子、鏡頭有沒有連上。偵測不到 JetBot 就用假馬達（只顯示數值）；鏡頭打不開就顯示「未連接鏡頭」 |
| 畫面 | 模型實際看到的影像（紅色車道線）、綠色預測點、上方橘色轉向條 |
| 輸入干擾 | 柏林噪聲、壓縮失真（按鈕） |
| 控制參數（全部是按鈕或滑桿） | 轉向角公式、max / min speed、kp、ki、kd、steering bias、轉彎減速、角度平滑 α、馬達變化率、馬達上限、內側輪可反轉、安全逾時 |
| 即時讀數 | x、y、angle、steering、實際 speed、左右馬達、P / I / D 項、FPS、前處理＋推論延遲、控制狀態 |
| 運轉 | **開始行駛／停止**（同一個按鈕：綠色「開始行駛」↔ 紅色「停止」）、錄製 CSV |

**安全機制**（開車時才會停車；推論不會停）：

| 狀況 | 結果 |
|---|---|
| 模型輸出 NaN / inf、超出畫面、目標點貼到車頭 | 停車 |
| 相機超過「安全逾時」秒數沒有新畫面 | 停車 |
| 推論或相機卡住，超過安全逾時都沒有成功的控制（看門狗執行緒） | 停車 |
| 推論出錯 | 停車並卸載模型，錯誤顯示在模型資訊列 |
| 切換模型 | 先停車 |
| sanity 比對沒通過、未載入模型、未連接鏡頭 | 按「開始行駛」不會動，會顯示原因 |

停車時按鈕會自動變回綠色，紅字顯示原因。

**沒有車也能看介面**：PC 上裝好 `jupyter`、`ipywidgets` 和 `requirements-pc.txt` 的套件後，開 `live_demo.ipynb` 執行全部 cell 即可。PC 沒有 TensorRT，下拉選單會是空的或載入失敗，這是正常的；要看推論畫面的話，第 1 節可以改 `CAMERA_SOURCE = 'opencv'`（筆電 webcam）或 `'folder'`，`MODELS_DIR` 改指向 `models/`。

---

## DEMO 前（PC / Colab，有 GPU）

```bash
cd project4
python train.py --prune 0.3 0.5                                   # 無 augmentation
python train.py --general --prune 0.3 0.5                         # 一般做法
python train.py --general --perlin --prune 0.3 0.5                # + 柏林噪聲
python train.py --general --perlin --compression --prune 0.3 0.5  # + 壓縮失真

# MobileNetV2：同樣的指令加 --arch mobilenet_v2
python train.py --arch mobilenet_v2 --general --perlin --prune 0.3 0.5

# 報告要比較方法時，用多個 seed 看平均和標準差
python train.py --general --perlin --prune 0.3 --seeds 0 1 2
```

資料不在預設的 `data/dataset_xy_2026-10-21/dataset_xy` 時，每行加上 `--data <資料夾>`。

用隊友分好的資料（`data/splits/{train,validation,test}/`）時，每行加上 `--data data/splits`。每個模型存檔前會印出 test 的 MSE 和像素誤差，並寫進 `.json`；`live_demo` 的模型資訊列也會顯示。比較不同模型時請看 test，因為 validation 已經被用來挑 epoch，會偏樂觀。

看 log：如果剪枝 50% 的 finetune 後明顯比 baseline 差，DEMO 就用 30% 的版本。

前處理固定會做，訓練前可以先檢查車道線標得對不對：

```bash
python preprocessing.py review --data <資料夾>     # 看 label_review.jpg
```

## DEMO 前一天（Nano）：轉 TensorRT

轉換必須在板子上做，每個模型要幾分鐘，**不要留到 DEMO 當天**。

1. 把以下檔案同步到板子的同一個資料夾（例如 `~/project4/`）：
   - `*.py`、`live_demo.ipynb`
   - PC 上 `models/` 裡要用的 `.pth` 和同名 `.json`（**兩個都要**：`.json` 裡有 SHA256 和比對畫面），放到板子的 `models/`
2. 在板子的終端機：

   ```bash
   cd ~/project4
   python3 convert_trt.py
   ```

   每個模型會顯示 `PASSED` / `FAILED`、比對的最大差和延遲，最後列出總結。轉好的模型在 `models_trt/`。
   - `FAILED`：轉換後的輸出和 PC 上差太多，`live_demo` 會禁止用它開車。可以試 `python3 convert_trt.py --method trtexec --force models/X.pth`（需要 PC 上用 `--onnx` 匯出的 `X.onnx`）。
   - `ERROR: SHA256 does not match`：`.pth` 沒傳完整或配錯 `.json`，重新上傳。
3. 開 `live_demo.ipynb` 由上往下執行，選一個模型，確認 sanity 比對通過、綠點正常。
4. 第 5 節的 `benchmark()` 可以先跑一次，記下各模型的延遲給報告用。

## DEMO 當天：操作車子

### 1. 開機連線

1. 車子開機，筆電連上板子的 AP，打開 Jupyter（port 8888）。
2. **把其他 notebook 的 kernel 都關掉**。相機同時只能被一個程式佔用。
   - 用 Running 分頁或 Kernel → Shut Down。

### 2. 啟動介面

1. 開 `live_demo.ipynb`，確認第 1 節的設定是 `CAMERA_SOURCE = 'jetbot'`。車子和鏡頭會自動偵測，介面上會顯示「已連接」。
2. 由上往下執行第 1 到第 6 節，用 Run All 也可以。第 3 節會出現介面，第 6 節是「結束」按鈕，**不會自動執行**。

### 3. 架高測試（輪子離地）

1. 下拉選單選要 DEMO 的模型。載入完成就開始推論，確認 **sanity 比對通過**（綠字）。
2. 把車子放在跑道上不同位置，確認綠點落在該去的方向，FPS 和延遲合理。
3. 把車架高讓輪子離地，按 **開始行駛**（按鈕變紅色「停止」）。
   - 兩輪都會轉，起步會慢慢加速（馬達變化率）。目標在右邊時左輪比較快，目標在左邊時右輪比較快。
   - 方向反了就先停下來檢查接線或程式。
4. 按 **停止**（按鈕變回綠色）。推論會繼續跑。

### 4. 放到跑道上

照投影片 P.72 的原則：先用低速，一次只改一個參數。預設值是隊友的起始值：max speed 0.20、min speed 0.10、kp 0.18、kd 0.008、角度平滑 0.35、馬達變化率 0.6、馬達上限 0.4。

1. 把車放在直線段，按 **開始行駛**。
2. **先調 kp**：轉彎轉不過去就加大；左右擺動就減小。
3. 擺動還是明顯時：
   - 加一點 **kd**（每秒的變化率，從 0.005 開始慢慢加），或
   - 把 **角度平滑 α** 調小（更平滑，但反應變慢）。
4. 直線上一直往同一邊偏，就調 **steering bias**（-0.3~0.3），補償左右馬達的差異；長時間偏一邊也可以加一點 **ki**。
5. 穩定之後再慢慢提高 max speed。速度變快通常要重新微調 kp / kd。必要時把 **馬達上限** 調高。
6. **髮夾彎轉不過去**：
   - 先加大 **轉彎減速**，轉彎時會自動變慢（但不低於 min speed），直線不受影響。
   - 還是轉不過去，再開 **內側輪可反轉**，讓車子可以原地轉。
   - 根本的解法是補拍髮夾彎的資料（入彎、彎中、出彎各角度 30~50 張，目標點放近），再重新訓練。
7. 想比較兩種轉向角公式，直接按 **轉向角公式** 的按鈕切換，不用重新訓練。

### 5. 比較不同變因

- 從下拉選單換模型時，車子會**自動停車**。
  - 換完會自動清掉相機堆積的舊畫面（畫面會停 1~2 秒），之後畫面就是即時的。
  - 換完先看 sanity 比對和綠點正不正常，再按「開始行駛」。
- 按「輸入干擾」按鈕可以現場示範：有柏林噪聲 / 壓縮 augmentation 的模型在干擾下比較穩。
- 按 **錄製 CSV**，每一幀的模型、x、y、角度、P / I / D、左右馬達、狀態、延遲和所有控制參數都會寫進 `runs/drive_*.csv`，報告可以用來畫圖。

### 6. 停車與結束

- 隨時按 **停止**：立刻停車，推論繼續跑。
- 狀況不對時，直接把車拿起來再按停止。
  - Jupyter 的 interrupt（■）**停不了背景推論迴圈**，要用「停止」按鈕。
- 結束時按第 6 節的 **結束：停車並釋放相機** 按鈕。下次要開別的 notebook 前一定要按。
  - 按了之後相機就釋放了，要再用必須 Restart Kernel 再重新執行。

### 報告要留的證據（投影片 P.79、P.80）

- [ ] 各模型 `.json`：epoch 數、訓練 / 驗證 / 測試張數、loss 曲線、像素誤差、MACs / params
- [ ] 多 seed 的 `<名稱>_seeds.json`：平均 ± 標準差
- [ ] `models_trt/*.json` 的 `trt_validation`：轉換後比對結果和模型延遲
- [ ] `benchmark()` 的延遲 / FPS，所有模型在相同條件下量
- [ ] `runs/drive_*.csv` 和行駛影片：跑幾圈、有沒有出界、擺動情況
- [ ] 最後用的控制參數和轉向角公式，以及調整過程

## 疑難排解

| 狀況 | 處理 |
|---|---|
| 下拉選單是空的 | `models_trt/` 裡沒有模型：先在車上跑 `python3 convert_trt.py`，再按「重新掃描」 |
| 載入失敗：SHA256 does not match | `.pth` 和 `.json` 對不上（沒傳完整或配錯），重新上傳再轉換 |
| sanity 比對沒通過 | TRT 轉換後的輸出和 PC 差太多，換 `--method trtexec` 重轉，或用另一個模型 |
| 換模型之後畫面一直延遲 | 確認第 1 節 `FLUSH_ON_SWITCH = True`。還是延遲的話，改用 `CAMERA_SOURCE = 'csi'`（Restart Kernel 後重跑） |
| 畫面沒更新 | **先看模型選單下方有沒有紅字**。推論出錯時會停車、卸載模型，並把錯誤和 traceback 顯示在那裡 |
| 顯示「未連接鏡頭」 | 其他 notebook 還佔著相機：關掉所有 kernel 再試，還不行就 `sudo systemctl restart nvargus-daemon`，然後 Restart Kernel |
| 車子一直被「逾時」停下 | 推論太慢或相機卡住。看 FPS 和延遲；確定沒問題的話可以把「安全逾時」調大一點 |
| 綠點亂跳或完全不對 | 是模型或資料的問題，不是控制的問題（投影片 P.61）。換一個模型比較看看 |
| 綠點合理但車子擺動 | 是控制的問題：降低 kp 或 max speed，再加一點 kd 或調小角度平滑 α |
| `No module named torch2trt` | 在 Nano 上確認有裝課程的 torch2trt |

## 已知風險（沙盒內無法驗證）

- **TensorRT 轉換和載入都還沒在板子上跑過**（沙盒沒有 GPU，`convert_trt.py` 只用假的 torch2trt 測過流程）。
  - 舊版 torch2trt 可能不支援某些運算（例如 `torch.flatten`），轉換會報錯；這時改用 `--method trtexec`。
  - 舊版 torch2trt 的 `TRTModule` 用 implicit batch 的 `execute_async`，碰到 trtexec 建的 explicit batch engine 可能報錯。
- Nano 上的 Python 3.6 / 舊版 ipywidgets 只做過靜態語法檢查，介面排版（例如按鈕圖示）可能和 PC 上不同。
- 換模型時的 flush（重啟 jetbot 相機管線）和 `CAMERA_SOURCE = 'csi'` 都還沒在板子上試過。flush 有問題就把 `FLUSH_ON_SWITCH` 改成 `False`。
- 控制預設值來自隊友，還沒在跑道上調過。
- 和投影片流程不同的地方，報告要寫清楚，DEMO 前建議先問助教：
  - 訓練和剪枝在 PC 上做，不是在 JetBot 上（P.79）。
  - 轉向角公式預設用隊友的版本，不是投影片的（介面上可以切回投影片公式）。
