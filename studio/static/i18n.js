// English texts from the server / task declarations -> Traditional Chinese.
// Python code stays English (ASCII rule); the page translates what it shows.
// Texts without an entry are shown in English.
var T = {
  // Groups of parameters
  "Safety": "安全",
  "Control": "控制",
  "Camera": "相機",
  "Model": "模型",
  "Display": "顯示",
  "Drive": "行駛",
  // Shared safety parameter
  "Link timeout (s)": "斷線容忍時間（秒）",
  "Stop the motors when no open page has answered for this long (page closed, Wi-Fi lost, web process crashed). Longer values tolerate a weak Wi-Fi but stop later.":
    "所有開著的頁面都超過這麼久沒有回應（關閉分頁、Wi-Fi 斷線、網頁行程當掉）就停馬達。設長一點比較能容忍 Wi-Fi 不穩，但停得比較晚。",
  // drive_check
  "Camera and motor check": "相機與馬達檢查",
  "Live camera picture and hold-to-drive buttons. Lift the wheels first.": "即時相機畫面＋按住才會動的馬達按鈕。請先把車子架高。",
  "Max motor power": "馬達最大值",
  "Motor value used by the buttons (0..1).": "按鈕送出的馬達值（0~1）。",
  "Hold time (s)": "按住有效時間（秒）",
  "Motors stop when no drive command arrived for this long (release the button = stop).": "超過這麼久沒收到行駛指令就停馬達（放開按鈕＝停車）。",
  "Draw motor values on the picture": "在畫面上顯示馬達值",
  // navigate / search
  "Avoid obstacles and drive to a clicked target": "避障＋點選目標前進",
  "Click a point in the picture; the robot drives there and avoids obstacles. Every click and emergency stop becomes a training label (interactive imitation learning).": "在畫面上點一個點，車子會避開障礙開過去。每次點選和緊急停止都會存成訓練標註（互動式模仿學習）。",
  "Object search and 3D map": "目標物搜尋＋3D 地圖",
  "Upload photos of an object; the robot explores, builds a map and marks the object, the start point and its route.": "上傳目標物照片，車子自己探索、建地圖，並標出目標物、起點和行駛路線。",
  // lane_follow
  "Lane following": "車道線跟隨",
  "Follow the taped road with a trained model (project4). Datasets, training, model library, TensorRT conversion and driving.": "用訓練好的模型沿著膠帶道路行駛（project4）。資料集、訓練、模型庫、TensorRT 轉換和行駛。",
  "Model folder": "模型資料夾", "Auto (car: models_trt, PC: models)": "自動（車上：models_trt、電腦：models）",
  "models_trt (TensorRT)": "models_trt（TensorRT）", "models (PyTorch)": "models（PyTorch）",
  "Flush camera after loading": "載入後清空相機佇列", "Drop frames queued while a model loads (jetbot source restarts its pipeline).": "丟掉載入模型期間堆積的畫面（jetbot 來源會重啟相機管線）。",
  "Torch CPU threads": "torch CPU 執行緒數", "0 = torch default. 1 can make the control period steadier (see HANDOFF M0).": "0＝torch 預設。設 1 可能讓控制週期比較穩（見 HANDOFF M0）。",
  "Steering formula": "轉向角公式", "Teammate formula": "隊友公式", "Slides formula": "投影片公式",
  "Max speed": "max speed", "Min speed": "min speed", "kd (per second)": "kd（每秒）", "Steering bias": "steering bias",
  "Turn slowdown": "轉彎減速", "Angle smoothing alpha": "角度平滑 α", "Motor slew per second": "馬達變化率／秒",
  "Motor limit": "馬達上限", "Inner wheel may reverse": "內側輪可反轉",
  "Inference / camera timeout (s)": "推論／相機逾時（秒）", "Stop when no successful control step happened for this long (model or camera stalled).": "超過這麼久沒有成功的控制步驟（模型或相機卡住）就停車。",
  "Data": "資料", "Network": "網路", "Training": "訓練", "Compression": "壓縮", "Output": "輸出",
  "Dataset": "資料集", "Folder below project4/data (flat, or with train/ validation/ test/).": "project4/data 底下的資料夾（單一資料夾，或含 train/ validation/ test/）。",
  "Split": "切分方式", "Random": "隨機", "By recording session": "依拍攝段落",
  "Only for flat datasets; split datasets use their folders as they are.": "只對單一資料夾的資料集有用；已切好的資料集直接用它的資料夾。",
  "Validation share": "驗證集比例", "Split seed": "切分 seed", "Mark lane lines": "標車道線",
  "Off = train on plain images (train.py --no-lanes, control group).": "關＝用原圖訓練（train.py --no-lanes，對照組）。",
  "Augmentation": "資料增強", "All combinations of at most 2 of 7 methods (29 per image).": "7 種方法中最多疊 2 種的所有組合（每張 29 種）。",
  "Loss": "損失函數", "Epochs": "epochs", "Learning rate": "學習率", "Batch size": "batch 大小",
  "Halved automatically and retried when the GPU runs out of memory.": "GPU 記憶體不足時自動減半重試。",
  "Early stop": "提早停止", "Seeds": "seeds", "Space-separated, e.g. '17 29 43' (mean +- std over seeds).": "用空白分開，例如「17 29 43」（算平均 ± 標準差）。",
  "Data loader workers": "DataLoader workers", "Pruning ratios": "剪枝比例", "Space-separated, e.g. '0.3 0.5'; empty = no pruning.": "用空白分開，例如「0.3 0.5」；空白＝不剪枝。",
  "Fine-tune epochs": "finetune epochs", "Fine-tune learning rate": "finetune 學習率", "Also export ONNX": "也匯出 ONNX",
  "Needed for TensorRT conversion with trtexec.": "用 trtexec 轉 TensorRT 時需要。", "Run name": "名稱",
  "Prefix of the output files; empty = automatic.": "輸出檔名的前綴；空白＝自動。",
  "choose a dataset": "請選資料集", "at least one seed": "至少一個 seed",
  // navigate
  "Mode": "模式", "Rule-based (free space + tracker)": "規則版（空地估計＋目標追蹤）", "Learned model": "學習版（模型）",
  "Free space": "空地估計", "Planner": "規劃", "Labels": "標註", "Sectors": "扇區數",
  "Horizon (fraction of height)": "地平線（畫面高度比例）", "Only the picture below this line is analysed.": "只分析這條線以下的畫面。",
  "Floor similarity threshold": "地板相似度門檻", "White tape counts as floor": "白膠帶算地板",
  "Free distance to drive into": "可以開進去的空地距離", "Stop distance (obstacle ahead)": "停車距離（前方障礙）",
  "Last protection, independent of any model: stop when the sectors ahead are blocked closer.": "最後一道保護，不依賴模型：正前方的扇區比這個距離還近就停車。",
  "Arrived when the target is this low": "目標低於這個位置就算到達", "Record 'free' pictures while driving": "行駛中記錄「正常行駛」畫面",
  "Loop / camera timeout (s)": "迴圈／相機逾時（秒）",
  // search
  "Search": "搜尋", "Map": "地圖", "Target name": "目標物名稱", "Name of a registered target (see 'Targets' on this page).": "已登錄的目標物名稱（見本頁「目標物登錄」）。",
  "Time limit (s)": "時間上限（秒）", "Hits in a row to confirm": "連續命中幾幀才算找到",
  "Camera horizontal field of view (deg)": "相機水平視角（度）", "Ignored when a calibration file is given.": "有校正檔時不使用。",
  "Camera height above the floor (m)": "相機離地高度（公尺）", "Sets the map's scale: measure it.": "決定地圖的比例，請實際量。",
  "Camera tilt down (deg)": "相機往下傾斜（度）", "Calibration file": "校正檔",
  "JSON from python3 -m studio.tasks.search.calibrate (fisheye K and D).": "python3 -m studio.tasks.search.calibrate 產生的 JSON（魚眼 K、D）。",
  "Map size (m)": "地圖大小（公尺）", "Map cell size (m)": "地圖格子大小（公尺）", "Keyframe every (m)": "每走幾公尺存一張關鍵幀",
  "Turn-on-the-spot power": "原地轉向的馬達值",
  "Free distance to drive into (m)": "可以開進去的空地距離（公尺）", "Stop distance (m, obstacle ahead)": "停車距離（公尺，前方障礙）",
  "Last protection, independent of any model: stop when the floor straight ahead ends closer than this.": "最後一道保護，不依賴模型：正前方的地板在這個距離內就結束時停車。",
  "Turns picture rows into metres: measure it.": "用來把畫面的列換算成公尺，請實際量。",
  // Runner / server messages
  "task is not running": "任務沒有在執行",
  "link_down": "沒有開著的頁面在回應，不能開動（請確認網路）",
  "invalid parameters": "參數不正確",
  "unknown parameter": "沒有這個參數",
  "cannot be changed while the task is running": "任務執行中不能修改",
  "must be a number": "必須是數字",
  "must be a whole number": "必須是整數",
  "must be true or false": "必須是開或關",
  "this task is still in development": "這個任務還在開發中",
  "login required": "請重新登入"
};

// Translate one text (unknown texts stay as they are)
function t(text) {
  // Dictionary entry or the original
  return (text in T) ? T[text] : text;
}

// Translate every element with class "t" (rendered by the server in English)
function translateAll(root) {
  // Every marked element
  (root || document).querySelectorAll(".t").forEach(function (el) {
    // Original text, trimmed
    var s = el.textContent.trim();
    // Replace when there is a translation
    if (s in T) el.textContent = T[s];
  });
}
