// Home page: running task, background jobs, disk quotas (from /api/status).

// Escape text for HTML
function esc(s) {
  // Replace the special characters
  return String(s === null || s === undefined ? "" : s).replace(/[&<>"]/g, function (c) {
    // Entity for each
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
  });
}

// Readable job states
var JOB_STATES = { queued: "排隊中", running: "執行中", done: "完成", failed: "失敗", cancelled: "已取消", lost: "中斷（伺服器重啟）" };

// Cancel a job and refresh
function cancelJob(id) {
  // Ask the server
  api("POST", "/api/jobs/" + id + "/cancel").then(refreshStatus);
}

// Update the page from a status
window.onStatus = function (st) {
  // Running task box
  var box = document.getElementById("running");
  // Runner state
  var r = st.runner;
  // Show it while it is not idle
  if (r && (r.status === "running" || r.status === "starting" || r.status === "stopping" || r.status === "failed")) {
    // Visible
    box.classList.remove("hidden");
    // Title, state and link
    box.innerHTML = "<b>目前任務：</b><a href='/task/" + esc(r.task) + "'>" + esc(t(r.title)) + "</a> · " +
      esc(r.status) + (r.error ? "<pre class='err'>" + esc(r.error) + "</pre>" : "");
  } else {
    // Hidden when idle
    box.classList.add("hidden");
  }
  // Jobs table
  var rows = st.jobs.map(function (j) {
    // Progress summary
    var p = j.progress || {};
    // One row
    return "<tr><td>" + esc(j.name) + "</td><td>" + esc(JOB_STATES[j.status] || j.status) + "</td><td>" +
      esc(p.text || p.waiting || "") + "</td><td>" + new Date(j.created * 1000).toLocaleString() + "</td><td>" +
      (j.status === "queued" || j.status === "running" ? "<button onclick=\"cancelJob('" + esc(j.id) + "')\">取消</button>" : "") +
      "</td></tr>";
  });
  // Table or a note
  document.getElementById("jobs").innerHTML = rows.length ?
    "<table><tr><th>工作</th><th>狀態</th><th>進度</th><th>建立時間</th><th></th></tr>" + rows.join("") + "</table>" :
    "<span class='muted'>沒有背景工作</span>";
  // Quota rows
  var q = st.resources.quotas || {};
  // Names of the categories
  var names = { models: "模型", recordings: "資料集與錄影", maps: "地圖", logs: "日誌" };
  // One row per category
  document.getElementById("quotas").innerHTML = "<table>" + Object.keys(q).map(function (k) {
    // Used share
    var pct = 100 * q[k].used_mb / q[k].limit_mb;
    // Row with a warning colour when full
    return "<tr><td>" + (names[k] || k) + "</td><td" + (pct > 100 ? " class='err'" : "") + ">" +
      fmt(q[k].used_mb, 0) + " / " + q[k].limit_mb + " MB（" + fmt(pct, 0, "%") + "）</td></tr>";
  }).join("") + "</table>";
};
