// Task page: picture stream (ack backpressure), control socket (heartbeat,
// e-stop, commands, live parameters), start / stop, presets, telemetry, events.
(function () {
  // Task declaration rendered by the server
  var task = window.STUDIO_TASK;
  // ws:// or wss:// matching the page
  var wsBase = (location.protocol === "https:" ? "wss://" : "ws://") + location.host;
  // Elements used below
  var $ = function (id) { return document.getElementById(id); };
  // Current runner state from the control socket
  var runner = null;

  // ---------------------------------------------------------------- form
  // Live edits are sent at most every 150 ms per parameter
  var pending = {}, timer = null;
  // Parameter form; live edits go to the running task
  var form = buildForm($("params"), task.params, function (name, value, valid) {
    // Only valid live values while running
    if (!valid || !running() || !form.isLive(name)) return;
    // Collect the change
    pending[name] = value;
    // Send soon (coalesces slider drags)
    if (!timer) timer = setTimeout(function () {
      // Send everything collected
      control.send({ type: "params", values: pending });
      // Start collecting again
      pending = {}; timer = null;
    }, 150);
  });

  // Whether the task runs right now
  function running() { return runner && runner.status === "running"; }

  // ---------------------------------------------------------------- stream
  // Picture element and its message overlay
  var img = $("video"), msg = $("video-msg");
  // Frames shown in the current second, and the last FPS
  var nFrames = 0, fps = 0, lastMeta = {};
  // Count FPS once per second
  setInterval(function () { fps = nFrames; nFrames = 0; }, 1000);
  // Open the stream socket (reconnects when closed)
  function openStream() {
    // New socket
    var ws = new WebSocket(wsBase + "/ws/stream/" + task.id);
    // Binary frames as ArrayBuffer
    ws.binaryType = "arraybuffer";
    // One frame arrived
    ws.onmessage = function (e) {
      // Header length (big endian)
      var view = new DataView(e.data);
      // JSON header length
      var hlen = view.getUint32(0);
      // Parse the header
      var header = JSON.parse(new TextDecoder().decode(new Uint8Array(e.data, 4, hlen)));
      // JPEG bytes as a blob URL
      var url = URL.createObjectURL(new Blob([new Uint8Array(e.data, 4 + hlen)], { type: "image/jpeg" }));
      // Once displayed: free the old URL and acknowledge (the server sends the next frame only then)
      img.onload = img.onerror = function () {
        // Free the blob
        URL.revokeObjectURL(url);
        // Ready for the next frame
        if (ws.readyState === 1) ws.send(JSON.stringify({ type: "ack", seq: header.seq }));
      };
      // Show it
      img.src = url;
      // Hide the overlay
      msg.classList.add("hidden");
      // Count it
      nFrames += 1;
      // Keep the metadata for the stats line
      lastMeta = header;
      // Delay from capture to display, FPS and quality
      $("video-stats").textContent = "畫面 " + fps + " FPS · JPEG " + header.quality +
        " · 延遲 " + Math.max(0, Date.now() / 1000 - header.t).toFixed(2) + " 秒（含時鐘差）";
      // Task page hook
      if (window.onFrame) window.onFrame(header);
    };
    // Reconnect after a short pause
    ws.onclose = function () { setTimeout(openStream, 1000); };
  }

  // ---------------------------------------------------------------- control
  // Control socket wrapper with reconnect and heartbeat
  var control = {
    // Current socket
    ws: null,
    // Pending command callbacks by id
    waiting: {},
    // Next command id
    nextId: 1,
    // Send JSON if connected
    send: function (m) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(m)); },
    // Run a task command; resolves with the reply
    command: function (name, args) {
      // This command's id
      var id = this.nextId++;
      // Remember the callback
      var self = this;
      // Promise resolved by the reply
      return new Promise(function (resolve) {
        // Store it
        self.waiting[id] = resolve;
        // Send the command
        self.send({ type: "command", id: id, name: name, args: args || {} });
      });
    }
  };
  // Open the control socket
  function openControl() {
    // New socket
    var ws = new WebSocket(wsBase + "/ws/control/" + task.id);
    // Use it
    control.ws = ws;
    // Connected
    ws.onopen = function () { $("link").textContent = "頁面已連線"; $("link").className = "badge ok"; };
    // Lost: show it and reconnect
    ws.onclose = function () {
      // Badge
      $("link").textContent = "頁面斷線，重新連線中…"; $("link").className = "badge bad";
      // Retry
      setTimeout(openControl, 1000);
    };
    // Messages from the server
    ws.onmessage = function (e) {
      // Parse
      var m = JSON.parse(e.data);
      // Runner state
      if (m.type === "state") showState(m);
      // Telemetry from the task
      else if (m.type === "telemetry") showTelemetry(m);
      // Event
      else if (m.type === "event") addEvent(m.event, m.replay);
      // Command reply
      else if (m.type === "reply" && control.waiting[m.id]) { control.waiting[m.id](m); delete control.waiting[m.id]; }
      // Live parameter result: show refused values
      else if (m.type === "params_result") form.setErrors(m.errors);
    };
  }
  // Heartbeat every 100 ms: keeps the motors allowed while this page is open and connected
  setInterval(function () { control.send({ type: "hb" }); }, 100);
  // Emergency stop through the control socket first (fastest path)
  estopHook = function () { control.send({ type: "estop" }); };
  // Expose for task pages (page.html scripts)
  window.studio = { control: control, form: form, task: task, running: running };

  // ---------------------------------------------------------------- state
  // Readable runner states
  var STATES = { starting: "啟動中…", running: "執行中", stopping: "停止中…", stopped: "已停止", failed: "失敗" };
  // Show the runner state
  function showState(m) {
    // Remember it
    runner = m.runner;
    // Badge text and colour
    var s = runner ? runner.status : null;
    // Badge
    $("state").textContent = s ? (STATES[s] || s) : "未啟動";
    // Colour by state
    $("state").className = "badge " + (s === "running" ? "ok" : s === "failed" ? "bad" : "");
    // Buttons
    if ($("start")) { $("start").disabled = !!s && s !== "stopped" && s !== "failed"; $("stop").disabled = s !== "running"; }
    // Fixed parameters locked while running
    form.setRunning(running());
    // A page opened while the task runs shows the values it runs with (once per run)
    if (running() && showState.paramsFor !== runner.started) { form.setValues(runner.params); showState.paramsFor = runner.started; }
    // Overlay text when there is no picture
    if (!running()) { msg.classList.remove("hidden"); msg.textContent = s === "starting" ? "啟動中…（車上載入 torch 約需 10~30 秒）" : "任務未啟動"; }
    // Another task holds the camera / motors
    var o = $("other");
    // Show the conflict
    if (m.other) { o.classList.remove("hidden"); o.innerHTML = "<div>相機和馬達正被「" + t(m.other.title) + "」使用中。</div>"; }
    // No conflict
    else o.classList.add("hidden");
    // Failure reason
    if (runner && runner.status === "failed" && runner.error && !showState.shown) {
      // Show it once as an event
      addEvent({ kind: "runner_failed", error: runner.error, t: Date.now() / 1000 });
      // Only once
      showState.shown = true;
    }
    // Reset when it runs again
    if (running()) showState.shown = false;
  }
  // Show telemetry as indented JSON (task pages may draw it nicer via onTelemetry)
  function showTelemetry(m) {
    // Link badge from the runner's point of view
    if (m.link === false && running()) { $("link").textContent = "車子沒收到心跳"; $("link").className = "badge warn"; }
    // Raw view
    $("telemetry").textContent = JSON.stringify(m.data, null, 1);
    // Task page hook
    if (window.onTelemetry) window.onTelemetry(m.data, m);
  }
  // Readable event texts
  var KINDS = { runner_ready: "任務已啟動", runner_stopped: "任務已停止", runner_died: "任務行程意外結束",
                runner_failed: "啟動失敗", safety_stop: "安全停車", error: "錯誤" };
  // Add one event to the list (newest on top)
  function addEvent(ev, replay) {
    // List item
    var li = document.createElement("li");
    // Faults in red
    var fault = ev.kind === "safety_stop" || ev.kind === "runner_died" || ev.kind === "runner_failed" ||
      ev.kind === "error" || ev.fault;
    // Class
    if (fault) li.className = "fault";
    // Time, kind and details
    var details = Object.keys(ev).filter(function (k) { return k !== "kind" && k !== "t"; })
      .map(function (k) { return k + "=" + (typeof ev[k] === "object" ? JSON.stringify(ev[k]) : ev[k]); }).join(" ");
    // Text
    li.textContent = new Date((ev.t || Date.now() / 1000) * 1000).toLocaleTimeString() + " " +
      (KINDS[ev.kind] || t(ev.kind)) + (details ? "：" + details : "");
    // Newest first
    $("events").insertBefore(li, $("events").firstChild);
    // Task page hook
    if (window.onEvent) window.onEvent(ev, replay);
  }

  // ---------------------------------------------------------------- buttons
  // Start the task with the form's values
  if ($("start")) $("start").onclick = function () {
    // Values and client-side validity
    var v = form.values();
    // Do not send invalid values
    if (!v.valid) { addEvent({ kind: "error", message: "參數有錯，請先修正紅框的欄位" }); return; }
    // Disable while starting
    $("start").disabled = true;
    // Ask the server (may take a while on the car)
    api("POST", "/api/tasks/" + task.id + "/start", { values: v.values }).then(function (r) {
      // Field errors from the server
      if (r.data.errors) form.setErrors(r.data.errors);
      // Failure reason
      if (r.status !== 200) addEvent({ kind: "runner_failed", error: t(r.data.error) });
      // Re-enable on failure
      $("start").disabled = r.status === 200;
    });
  };
  // Stop the task
  if ($("stop")) $("stop").onclick = function () { api("POST", "/api/tasks/" + task.id + "/stop"); };

  // ---------------------------------------------------------------- presets
  // Refill the preset list
  function loadPresetList(names) {
    // Select box
    var sel = $("preset");
    // Keep the first "(defaults)" entry
    sel.length = 1;
    // One option per preset
    names.forEach(function (n) { var o = document.createElement("option"); o.value = o.textContent = n; sel.appendChild(o); });
  }
  // Load the selected preset (or the defaults)
  $("preset-load").onclick = function () {
    // Selected name
    var name = $("preset").value;
    // Defaults
    if (!name) { var d = {}; task.params.forEach(function (p) { d[p.name] = p.default; }); form.setValues(d); return; }
    // Ask the server
    api("GET", "/api/tasks/" + task.id + "/presets/" + name).then(function (r) {
      // Fill the form
      if (r.status === 200) { form.setValues(r.data.values); form.setErrors(r.data.errors); }
    });
  };
  // Save the current values under a name
  $("preset-save").onclick = function () {
    // Name and values
    var name = $("preset-name").value.trim(), v = form.values();
    // Invalid values are not saved
    if (!v.valid) return;
    // Ask the server
    api("POST", "/api/tasks/" + task.id + "/presets", { name: name, values: v.values }).then(function (r) {
      // New list
      if (r.data.presets) loadPresetList(r.data.presets);
      // Errors (name or values)
      if (!r.data.saved) addEvent({ kind: "error", message: JSON.stringify(r.data.errors) });
      // Select the saved preset
      else $("preset").value = name;
    });
  };
  // Delete the selected preset
  $("preset-del").onclick = function () {
    // Selected name
    var name = $("preset").value;
    // Nothing selected
    if (!name || !confirm("刪除 preset「" + name + "」？")) return;
    // Ask the server
    api("DELETE", "/api/tasks/" + task.id + "/presets/" + name).then(function (r) { loadPresetList(r.data.presets || []); });
  };
  // Initial preset list
  api("GET", "/api/tasks/" + task.id + "/presets").then(function (r) { loadPresetList(r.data.presets || []); });

  // Start both sockets
  openStream();
  // Control socket
  openControl();
})();
