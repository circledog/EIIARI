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
