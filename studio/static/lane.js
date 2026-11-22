// Lane following sub-pages: datasets, training, model library (LANE_PAGE says which).
(function () {
  // Short element lookup
  var $ = function (id) { return document.getElementById(id); };
  // Escape text for HTML
  function esc(s) {
    // Replace the special characters
    return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, function (c) {
      // Entity for each
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  // Number with digits or a dash
  function num(v, d) { return v === null || v === undefined ? "-" : Number(v).toFixed(d); }

  // ------------------------------------------------------------ datasets
  // Datasets page
  function datasetsPage() {
    // Current dataset, split and page offset
    var ds = null, split = null, offset = 0, total = 0, LIMIT = 48;
    // Load the dataset table
    api("GET", "/api/lane/datasets").then(function (r) {
      // Rows
      var list = r.data.datasets || [];
      // No datasets
      if (!list.length) { $("ds-table").innerHTML = "<tr><td>" + esc(r.data.root) + " 裡沒有資料集</td></tr>"; return; }
      // Header and one row per dataset
      $("ds-table").innerHTML = "<tr><th>資料集</th><th>格式</th><th>張數</th><th></th></tr>" + list.map(function (d, i) {
        // Counts text
        var c = d.layout === "splits" ? "train " + d.counts.train + " / validation " + d.counts.validation + " / test " + d.counts.test : d.counts.all;
        // Row with a view button
        return "<tr><td>" + esc(d.id) + "</td><td>" + (d.layout === "splits" ? "已切好" : "單一資料夾") + "</td><td>" + c +
          "</td><td><button data-i='" + i + "'>瀏覽</button></td></tr>";
      }).join("");
      // View buttons
      $("ds-table").querySelectorAll("button").forEach(function (b) { b.onclick = function () { open(list[Number(b.dataset.i)]); }; });
    });
    // Open one dataset
    function open(d) {
      // Remember it
      ds = d; offset = 0;
      // Show the viewer
      $("ds-view").classList.remove("hidden");
      // Title
      $("ds-name").textContent = d.id;
      // Split choices
      var splits = d.layout === "splits" ? ["train", "validation", "test"] : ["all"];
      // Fill the select
      $("ds-split").innerHTML = splits.map(function (s) { return "<option>" + s + "</option>"; }).join("");
      // First split
      split = splits[0];
      // Clear the check result
      $("ds-check-result").textContent = "";
      // Load images
      page();
    }
    // Load the current page of images
    function page() {
      // Ask the server
      api("GET", "/api/lane/datasets/images?ds=" + encodeURIComponent(ds.id) + "&split=" + split + "&offset=" + offset + "&limit=" + LIMIT).then(function (r) {
        // Total for paging
        total = r.data.total;
        // Page text
        $("ds-page").textContent = " " + (total ? offset + 1 : 0) + "–" + Math.min(total, offset + LIMIT) + " / " + total + " ";
        // Raw or lane-marked
        var mode = $("ds-lanes").checked ? "lanes" : "raw";
        // Thumbnails
        $("ds-grid").innerHTML = r.data.items.map(function (it) {
          // Image URL
          var url = "/api/lane/image?ds=" + encodeURIComponent(ds.id) + "&split=" + split + "&name=" + encodeURIComponent(it.name) + "&mode=" + mode + "&size=160";
          // Figure with the label
          return "<figure><img loading='lazy' src='" + url + "'><figcaption>(" + it.x + ", " + it.y + ")</figcaption></figure>";
        }).join("");
      });
    }
    // Split changed
    $("ds-split").onchange = function () { split = this.value; offset = 0; page(); };
    // Lane marking toggled
    $("ds-lanes").onchange = page;
    // Previous page
    $("ds-prev").onclick = function () { if (offset > 0) { offset = Math.max(0, offset - LIMIT); page(); } };
    // Next page
    $("ds-next").onclick = function () { if (offset + LIMIT < total) { offset += LIMIT; page(); } };
    // Check for unreadable files
    $("ds-check").onclick = function () {
      // Progress text
      $("ds-check-result").textContent = "檢查中…";
      // Ask the server
      api("POST", "/api/lane/datasets/check", { ds: ds.id }).then(function (r) {
        // Result
        $("ds-check-result").textContent = "檢查 " + r.data.checked + " 張，壞檔 " + r.data.bad.length + " 張" +
          (r.data.bad.length ? "：" + r.data.bad.join(", ") : "");
      });
    };
  }

  // ------------------------------------------------------------ training
  // Draw validation px curves of a job into a canvas
  function drawCurves(canvas, curves) {
    // 2D context
    var g = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
    // Clear
    g.clearRect(0, 0, W, H);
    // Every point of every phase
    var all = [];
    // Collect them
    Object.keys(curves).forEach(function (k) { all = all.concat(curves[k]); });
    // Nothing yet
    if (!all.length) return;
    // Value range of val px and epochs
    var maxPx = Math.max.apply(null, all.map(function (p) { return p[3]; })), maxE = Math.max.apply(null, all.map(function (p) { return p[0]; }));
    // One colour per phase
    var colors = ["#1d6fdc", "#d9480f", "#2b8a3e", "#ae3ec9", "#e67700"];
    // Each phase
    Object.keys(curves).forEach(function (k, i) {
      // Line colour
      g.strokeStyle = colors[i % colors.length];
      // Start the path
      g.beginPath();
      // Each point
      curves[k].forEach(function (p, j) {
        // Canvas coordinates
        var x = 4 + (W - 8) * (p[0] - 1) / Math.max(1, maxE - 1), y = H - 4 - (H - 8) * p[3] / maxPx;
        // Line to the point
        if (j) g.lineTo(x, y); else g.moveTo(x, y);
      });
      // Draw it
      g.stroke();
    });
    // Scale label
    g.fillStyle = "#666";
    // Top value
    g.fillText("val " + maxPx.toFixed(0) + " px", 4, 10);
  }
  // Training page
  function trainingPage() {
    // The form (built from the server's schema)
    var form = null;
    // Load the schema
    api("GET", "/api/lane/train/schema").then(function (r) {
      // Build the form
      form = buildForm($("train-form"), r.data.schema);
      // No datasets
      if (!r.data.datasets.length) $("train-msg").textContent = "project4/data 裡沒有資料集";
    });
    // Start training
    $("train-start").onclick = function () {
      // Values and validity
      var v = form.values();
      // Invalid fields are outlined
      if (!v.valid) return;
      // Ask the server
      api("POST", "/api/lane/train", { values: v.values }).then(function (r) {
        // Field errors
        form.setErrors(r.data.errors || {});
        // Result text
        $("train-msg").textContent = r.status === 200 ? "已加入佇列：" + r.data.job.name : t(r.data.error);
        // Refresh the job list
        refresh();
      });
    };
    // Job states
    var STATES = { queued: "排隊中", running: "執行中", done: "完成", failed: "失敗", cancelled: "已取消", lost: "中斷" };
    // Log state: job id and offset
    var logJob = null, logOffset = 0;
    // Refresh the job list
    function refresh() {
      // Ask the server
      api("GET", "/api/jobs").then(function (r) {
        // Training jobs only
        var jobs = r.data.jobs.filter(function (j) { return j.kind === "train"; });
        // Cards with progress and curves
        $("train-jobs").innerHTML = jobs.length ? jobs.map(function (j) {
          // Progress
          var p = j.progress || {};
          // Card
          return "<div class='card'><b>" + esc(j.name) + "</b> · " + (STATES[j.status] || j.status) +
            (j.status === "running" || j.status === "queued" ? " <button data-cancel='" + esc(j.id) + "'>取消</button>" : "") +
            "<div class='small'>" + esc(p.text || p.waiting || "") + (p.best_px ? " · 最佳 val " + num(p.best_px, 1) + " px" : "") +
            (j.meta && j.meta.note ? "<br><span class='err'>" + esc(j.meta.note) + "</span>" : "") +
            "</div><canvas width='320' height='90' data-job='" + esc(j.id) + "'></canvas></div>";
        }).join("") : "<span class='muted'>還沒有訓練工作</span>";
        // Curves
        jobs.forEach(function (j) {
          // Canvas of the job
          var c = document.querySelector("canvas[data-job='" + j.id + "']");
          // Draw
          if (c) drawCurves(c, (j.progress || {}).curves || {});
        });
        // Cancel buttons
        document.querySelectorAll("[data-cancel]").forEach(function (b) {
          // Cancel and refresh
          b.onclick = function () { api("POST", "/api/jobs/" + b.dataset.cancel + "/cancel").then(refresh); };
        });
        // Log job choices
        var sel = $("train-log-job"), keep = sel.value;
        // One option per job
        sel.innerHTML = jobs.map(function (j) { return "<option value='" + esc(j.id) + "'>" + esc(j.name) + "</option>"; }).join("");
        // Keep the selection
        if (keep) sel.value = keep;
        // Follow the selected log
        tailLog();
      });
    }
    // Append new log text
    function tailLog() {
      // Selected job
      var id = $("train-log-job").value;
      // Nothing selected
      if (!id) return;
      // Another job: start from the beginning
      if (id !== logJob) { logJob = id; logOffset = 0; $("train-log").textContent = ""; }
      // Ask for the new part
      api("GET", "/api/jobs/" + id + "/log?offset=" + logOffset).then(function (r) {
        // Nothing new
        if (!r.data.text) return;
        // Append
        $("train-log").textContent += r.data.text;
        // Next offset
        logOffset = r.data.offset;
        // Scroll to the end
        $("train-log").scrollTop = $("train-log").scrollHeight;
      });
    }
    // Log job changed
    $("train-log-job").onchange = tailLog;
    // First load and every 2 s
    refresh();
    // Periodic refresh
    setInterval(refresh, 2000);
  }

  // ------------------------------------------------------------ models
  // Models page
  function modelsPage() {
    // Selected models for the comparison: "kind/name" -> info
    var picked = {}, role = null, data = null;
    // Table of one folder
    function table(list, kind) {
      // Empty folder
      if (!list.length) return "<span class='muted'>沒有模型</span>";
      // Header and rows
      return "<table><tr><th></th><th>檔名</th><th>說明</th><th>val px</th><th>test px（p95）</th><th>MB</th><th>sanity</th><th></th></tr>" + list.map(function (m) {
        // TensorRT validation result
        var tv = m.trt_validation || {};
        // Sanity column
        var san = kind === "trt" ? (tv.status ? tv.status + (tv.max_error_px !== undefined ? " " + num(tv.max_error_px, 2) + " px" : "") : "-") : (m.sanity_frames ? m.sanity_frames + " 張" : "沒有");
        // Row
        return "<tr><td><input type='checkbox' data-pick='" + kind + "/" + esc(m.name) + "'></td><td>" + esc(m.name) +
          (m.has_json ? "" : " <span class='err'>（缺 .json）</span>") + "</td><td>" + esc(m.label || "") + "</td><td>" +
          num(m.val_metrics && m.val_metrics.mean_px, 1) + "</td><td>" + num(m.test_metrics && m.test_metrics.mean_px, 1) +
          (m.test_metrics ? "（" + num(m.test_metrics.p95_px, 1) + "）" : "") + "</td><td>" + m.size_mb + "</td><td>" + esc(san) + "</td><td>" +
          (kind === "torch" && role === "car" ? "<button data-convert='" + esc(m.name) + "'>轉 TensorRT</button> " : "") +
          "<a href='/api/lane/models/file?kind=" + kind + "&name=" + encodeURIComponent(m.name) + "'>下載</a> " +
          (m.has_json ? "<a href='/api/lane/models/file?kind=" + kind + "&name=" + encodeURIComponent(m.name.replace(/\.pth$/, ".json")) + "'>.json</a> " : "") +
          "<button data-del='" + kind + "/" + esc(m.name) + "'>刪除</button></td></tr>";
      }).join("") + "</table>";
    }
    // Side-by-side comparison of the two picked models
    function compare() {
      // Picked infos
      var ms = Object.keys(picked).map(function (k) { return picked[k]; });
      // Need exactly two
      if (ms.length !== 2) { $("m-compare").innerHTML = "<span class='muted'>勾選兩個模型就會在這裡並排比較（目前 " + ms.length + " 個）。</span>"; return; }
      // Rows: label and getter
      var rows = [["檔名", function (m) { return m.name; }], ["說明", function (m) { return m.label; }], ["網路", function (m) { return m.arch; }],
        ["augmentation", function (m) { return m.augmentation; }], ["前處理", function (m) { return m.preprocessing; }],
        ["epochs / lr / batch", function (m) { return [m.epochs, m.lr, m.batch_size].join(" / "); }], ["seed", function (m) { return m.seed; }],
        ["val px", function (m) { return num(m.val_metrics && m.val_metrics.mean_px, 2); }],
        ["test px", function (m) { return num(m.test_metrics && m.test_metrics.mean_px, 2); }],
        ["test p95 px", function (m) { return num(m.test_metrics && m.test_metrics.p95_px, 2); }],
        ["test x / y px", function (m) { return m.test_metrics ? num(m.test_metrics.mae_x_px, 1) + " / " + num(m.test_metrics.mae_y_px, 1) : "-"; }],
        ["params (M)", function (m) { return num(m.params / 1e6, 2); }], ["MACs (G)", function (m) { return num(m.macs / 1e9, 2); }],
        ["TRT 延遲 ms", function (m) { var t = m.trt_validation || {}; return t.median_model_ms !== undefined ? num(t.median_model_ms, 1) : "-"; }],
        ["資料", function (m) { return m.data; }], ["程式版本", function (m) { return m.code_version && m.code_version.commit; }]];
      // Table
      $("m-compare").innerHTML = "<table>" + rows.map(function (r) {
        // Values of both
        var a = r[1](ms[0]), b = r[1](ms[1]);
        // Highlight differences
        return "<tr><th>" + r[0] + "</th><td" + (String(a) !== String(b) ? " style='background:#fff4d6'" : "") + ">" + esc(a) + "</td><td" +
          (String(a) !== String(b) ? " style='background:#fff4d6'" : "") + ">" + esc(b) + "</td></tr>";
      }).join("") + "</table>";
    }
    // Load both folders
    function refresh() {
      // Ask the server
      api("GET", "/api/lane/models").then(function (r) {
        // Remember
        data = r.data; role = r.data.role;
        // Tables
        $("m-torch").innerHTML = table(r.data.torch, "torch");
        // TensorRT table
        $("m-trt").innerHTML = table(r.data.trt, "trt");
        // Restore picks and wire the checkboxes
        document.querySelectorAll("[data-pick]").forEach(function (cb) {
          // Kind and name
          var parts = cb.dataset.pick.split("/"), list = data[parts[0]];
          // Still picked?
          cb.checked = !!picked[cb.dataset.pick];
          // Toggle
          cb.onchange = function () {
            // Info of that model
            var m = list.filter(function (x) { return x.name === parts[1]; })[0];
            // Add / remove
            if (cb.checked) picked[cb.dataset.pick] = m; else delete picked[cb.dataset.pick];
            // Update the comparison
            compare();
          };
        });
        // Delete buttons
        document.querySelectorAll("[data-del]").forEach(function (b) {
          // Ask, then delete
          b.onclick = function () {
            // Kind and name
            var parts = b.dataset.del.split("/");
            // Confirm
            if (!confirm("刪除 " + parts[1] + "（含 .json / .onnx）？")) return;
            // Ask the server
            api("POST", "/api/lane/models/delete", { kind: parts[0], name: parts[1] }).then(function () { delete picked[b.dataset.del]; refresh(); compare(); });
          };
        });
        // Convert buttons (car)
        document.querySelectorAll("[data-convert]").forEach(function (b) {
          // Queue a conversion
          b.onclick = function () {
            // trtexec needs the .onnx
            var method = confirm("用 torch2trt 轉換？（取消＝改用 trtexec，需要 .onnx）") ? "torch2trt" : "trtexec";
            // Ask the server
            api("POST", "/api/lane/models/convert", { name: b.dataset.convert, method: method, force: true }).then(function (r) {
              // Error
              if (r.status !== 200) alert(t(r.data.error)); else jobs();
            });
          };
        });
      });
    }
    // Conversion jobs
    function jobs() {
      // Ask the server
      api("GET", "/api/jobs").then(function (r) {
        // Conversion jobs only
        var list = r.data.jobs.filter(function (j) { return j.kind === "convert"; });
        // Table
        $("m-jobs").innerHTML = list.length ? "<table>" + list.map(function (j) {
          // Row
          return "<tr><td>" + esc(j.name) + "</td><td>" + esc(j.status) + "</td><td>" + esc((j.progress || {}).waiting || "") + "</td></tr>";
        }).join("") + "</table>" : "<span class='muted'>沒有轉換工作</span>";
        // A conversion finished: refresh the tables
        var done = list.filter(function (j) { return j.status === "done"; }).length;
        // Changed since last time
        if (done !== jobs.done) { jobs.done = done; refresh(); }
      });
    }
    // Upload
    $("up-go").onclick = function () {
      // Chosen files
      var files = $("up-files").files;
      // Nothing chosen
      if (!files.length) return;
      // Multipart body
      var fd = new FormData();
      // Every file
      for (var i = 0; i < files.length; i++) fd.append("files", files[i], files[i].name);
      // Progress text
      $("up-msg").textContent = "上傳中…";
      // Send with the XSRF header
      fetch("/api/lane/models/upload", { method: "POST", body: fd, headers: { "X-XSRFToken": cookie("_xsrf") }, credentials: "same-origin" })
        .then(function (r) { return r.json(); }).then(function (d) {
          // Result
          $("up-msg").textContent = d.saved ? "已存到 " + d.folder + "：" + d.saved.join(", ") : t(d.error);
          // Refresh the tables
          refresh();
        });
    };
    // First load
    refresh();
    // Conversion jobs now and every 3 s
    jobs();
    // Periodic
    setInterval(jobs, 3000);
  }

  // Run the page's code
  ({ datasets: datasetsPage, training: trainingPage, models: modelsPage })[LANE_PAGE]();
})();
