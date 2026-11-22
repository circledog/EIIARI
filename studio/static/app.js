// Shared page code: API helper with XSRF token, emergency stop, resource bar.

// Value of a cookie by name
function cookie(name) {
  // Find "name=value" in document.cookie
  var m = document.cookie.match("(^|;)\\s*" + name + "=([^;]*)");
  // Decoded value or empty
  return m ? decodeURIComponent(m[2]) : "";
}

// Call the JSON API; resolves to {status, data}
function api(method, url, body) {
  // Request options with the XSRF header tornado checks
  var opts = { method: method, headers: { "X-XSRFToken": cookie("_xsrf") }, credentials: "same-origin" };
  // JSON body when given
  if (body !== undefined) {
    // Content type
    opts.headers["Content-Type"] = "application/json";
    // Encoded body
    opts.body = JSON.stringify(body);
  }
  // Send and parse the JSON answer
  return fetch(url, opts).then(function (r) {
    // Logged out: back to the login page
    if (r.status === 401) location.href = "/login?next=" + encodeURIComponent(location.pathname);
    // Status and parsed body (empty object when not JSON)
    return r.json().catch(function () { return {}; }).then(function (d) { return { status: r.status, data: d }; });
  });
}

// Emergency stop: the page's own fast path (control socket) is set by task.js
var estopHook = null;

// Stop whatever runs, through every available channel
function emergencyStop() {
  // Fast path on task pages
  if (estopHook) estopHook();
  // HTTP path always (works on every page)
  api("POST", "/api/estop");
  // Visual feedback
  var b = document.getElementById("estop");
  // Flash the button text
  b.textContent = "已送出停止";
  // Restore after a second
  setTimeout(function () { b.textContent = "緊急停止"; }, 1000);
}

// Button click
document.getElementById("estop").addEventListener("click", emergencyStop);
// Space bar = emergency stop (but not while typing in a field)
document.addEventListener("keydown", function (e) {
  // Ignore keys typed into inputs
  var tag = (e.target.tagName || "").toLowerCase();
  // Space outside of text fields
  if (e.code === "Space" && tag !== "input" && tag !== "textarea" && tag !== "select") {
    // Do not scroll the page
    e.preventDefault();
    // Stop
    emergencyStop();
  }
});

// Format a number or a dash
function fmt(v, digits, unit) {
  // Unknown values
  if (v === null || v === undefined) return "-";
  // Fixed digits plus unit
  return v.toFixed(digits) + (unit || "");
}

// Latest status from the server (shared with page scripts)
var lastStatus = null;

// Refresh the resource bar and warnings
function refreshStatus() {
  // Ask the server
  api("GET", "/api/status").then(function (r) {
    // Ignore errors
    if (r.status !== 200) return;
    // Keep it for page scripts
    lastStatus = r.data;
    // Resources
    var s = r.data.resources;
    // One line: role, CPU, GPU, RAM, disk
    document.getElementById("res").textContent =
      (r.data.role === "car" ? "車上" : "電腦") + " · CPU " + fmt(s.cpu_percent, 0, "%") +
      " · GPU " + fmt(s.gpu_percent, 0, "%") +
      " · RAM 可用 " + fmt(s.mem_avail_mb, 0, " MB") +
      " · 硬碟剩 " + fmt(s.disk_free_mb / 1000, 1, " GB") +
      (s.runner_rss_mb ? " · 任務 " + fmt(s.runner_rss_mb, 0, " MB") : "");
    // Warning boxes
    var w = document.getElementById("warnings");
    // Rebuild them
    w.innerHTML = "";
    // One box per warning
    s.warnings.forEach(function (text) {
      // Box element
      var d = document.createElement("div");
      // Warning text
      d.textContent = "⚠ " + t(text);
      // Add it
      w.appendChild(d);
    });
    // Page-specific update
    if (window.onStatus) window.onStatus(r.data);
  });
}

// Translate server-rendered texts
translateAll();
// First status now, then every 2 seconds
refreshStatus();
// Periodic refresh
setInterval(refreshStatus, 2000);
