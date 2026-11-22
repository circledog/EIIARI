// Map viewer: 2D occupancy grid with route / start / object, and a small 3D point-cloud viewer
// (plain canvas, no external library: the robot's LAN may have no internet).
(function () {
  // Session id and base URL of its files
  var base = "/api/search/sessions/" + MAP_SID + "/";
  // Short element lookup
  var $ = function (id) { return document.getElementById(id); };
  // Fetch JSON (null when missing)
  function getJSON(name) { return fetch(base + name, { credentials: "same-origin" }).then(function (r) { return r.ok ? r.json() : null; }); }
  // Trajectory points [x, y] from the CSV
  function getTrajectory() {
    // Download the CSV
    return fetch(base + "trajectory.csv", { credentials: "same-origin" }).then(function (r) { return r.ok ? r.text() : ""; }).then(function (t) {
      // Lines without the header
      return t.trim().split("\n").slice(1).filter(Boolean).map(function (l) { var c = l.split(","); return [Number(c[1]), Number(c[2])]; });
    });
  }
  // Everything at once
  Promise.all([getJSON("summary.json"), getJSON("cloud.json"), getTrajectory()]).then(function (all) {
    // Parts
    var summary = all[0], cloud = all[1], path = all[2];
    // Object position, if found
    var obj = summary && summary.found && summary.found.object_xy;
    // Summary line
    if (summary) $("mv-summary").textContent = "已探索 " + summary.grid.explored_m2 + " m² · 關鍵幀 " + summary.keyframes + " · " + summary.duration_s + " 秒 · 里程計失敗 " + summary.odometry_lost + " 次";
    // Found photo
    if (summary && summary.found) { $("mv-found").classList.remove("hidden"); $("mv-found-info").textContent = "地圖位置 " + JSON.stringify(obj) + "（公尺），車子位置 " + JSON.stringify(summary.found.robot_pose); }
    // 2D view
    draw2d(summary, obj);
    // 3D view
    view3d(cloud, path, obj, summary);
  });

  // 2D: grid picture with the object marker on top
  function draw2d(summary, obj) {
    // Canvas
    var cv = $("mv-2d"), g = cv.getContext("2d"), img = new Image();
    // Once the grid picture is loaded
    img.onload = function () {
      // Canvas size = picture size
      cv.width = img.width; cv.height = img.height;
      // Draw the grid (route and start are part of the picture)
      g.drawImage(img, 0, 0);
      // Object marker
      if (obj && summary) {
        // Grid geometry
        var gr = summary.grid, s = gr.scale || 2;
        // Cell of the object (same formula as grid_map.cell)
        var cx = (obj[0] / gr.res + gr.n / 2) * s, cy = (-obj[1] / gr.res + gr.n / 2) * s;
        // Red dot
        g.fillStyle = "#e00"; g.beginPath(); g.arc(cx, cy, 6, 0, 7); g.fill();
      }
    };
    // Load it (no-cache: the search may still be running)
    img.src = base + "grid.png?t=" + Date.now();
  }

  // 3D: orbit camera, painter's algorithm
  function view3d(cloud, path, obj, summary) {
    // Canvas and context
    var cv = $("mv-3d"), g = cv.getContext("2d");
    // No reconstruction yet
    if (!cloud) { $("mv-3d-msg").textContent = "還沒有 3D 地圖：回到搜尋頁按「建 3D 地圖」（背景工作），完成後重新整理。"; }
    // Points and colours (empty without a cloud)
    var P = cloud ? cloud.points : [], C = cloud ? cloud.colors : [];
    // Info line
    if (cloud) $("mv-3d-msg").textContent = "地板 " + cloud.floor_points + " 點、結構 " + cloud.structure_points + " 點（顯示 " + P.length + " 點）";
    // Centre of the scene
    var cx = 0, cy = 0;
    // Mean of the route (or the points)
    var src = path.length ? path : P;
    // Sum
    src.forEach(function (p) { cx += p[0]; cy += p[1]; });
    // Mean
    if (src.length) { cx /= src.length; cy /= src.length; }
    // Orbit angles and distance
    var yaw = -0.8, pitch = 0.9, dist = 4;
    // Draw one view
    function draw() {
      // Canvas size follows the element
      cv.width = cv.clientWidth; cv.height = cv.clientHeight;
      // Clear
      g.fillStyle = "#111"; g.fillRect(0, 0, cv.width, cv.height);
      // View rotation
      var cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      // Focal length in pixels
      var f = cv.height * 1.2;
      // Project a world point -> [sx, sy, depth] or null
      function proj(x, y, z) {
        // Relative to the centre
        x -= cx; y -= cy;
        // Rotate about z (yaw)
        var xr = cyw * x - syw * y, yr = syw * x + cyw * y;
        // Tilt (pitch): camera looks down at the scene
        var yc = cp * yr - sp * z, zc = sp * yr + cp * z;
        // Depth along the view
        var d = dist + yc;
        // Behind the camera
        if (d < 0.05) return null;
        // Screen position
        return [cv.width / 2 + f * xr / d, cv.height / 2 - f * zc / d, d];
      }
      // Projected points with their colours
      var list = [];
      // Every point
      for (var i = 0; i < P.length; i++) {
        // Project
        var q = proj(P[i][0], P[i][1], P[i][2]);
        // Visible
        if (q) list.push([q[0], q[1], q[2], C[i]]);
      }
      // Far first (painter's algorithm)
      list.sort(function (a, b) { return b[2] - a[2]; });
      // Draw the points
      for (var k = 0; k < list.length; k++) {
        // Point
        var it = list[k], s = Math.max(1, 6 / it[2]);
        // Colour
        g.fillStyle = "rgb(" + it[3][0] + "," + it[3][1] + "," + it[3][2] + ")";
        // Square
        g.fillRect(it[0] - s / 2, it[1] - s / 2, s, s);
      }
      // Route in blue, slightly above the floor
      g.strokeStyle = "#39f"; g.lineWidth = 2; g.beginPath();
      // Every pose
      path.forEach(function (p, j) { var q = proj(p[0], p[1], 0.01); if (q) { if (j) g.lineTo(q[0], q[1]); else g.moveTo(q[0], q[1]); } });
      // Draw it
      g.stroke();
      // Start (green) and object (red) markers
      [[[0, 0], "#2c2", "起點"], [obj, "#f33", "目標物"]].forEach(function (m) {
        // No object
        if (!m[0]) return;
        // Pole from the floor to 15 cm
        var a = proj(m[0][0], m[0][1], 0), b = proj(m[0][0], m[0][1], 0.15);
        // Visible
        if (!a || !b) return;
        // Pole
        g.strokeStyle = m[1]; g.lineWidth = 3; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
        // Label
        g.fillStyle = m[1]; g.font = "13px sans-serif"; g.fillText(m[2], b[0] + 4, b[1]);
        // Remember the object's screen position for clicks
        if (m[2] === "目標物") view3d.objScreen = b;
      });
    }
    // Pointer state
    var drag = null, pinch = null;
    // Start dragging
    cv.addEventListener("pointerdown", function (e) { drag = [e.clientX, e.clientY, yaw, pitch]; cv.setPointerCapture(e.pointerId); });
    // Rotate while dragging
    cv.addEventListener("pointermove", function (e) {
      // Not dragging
      if (!drag) return;
      // New angles
      yaw = drag[2] - (e.clientX - drag[0]) * 0.01;
      // Pitch limited to look from above
      pitch = Math.max(0.1, Math.min(1.5, drag[3] + (e.clientY - drag[1]) * 0.01));
      // Redraw
      draw();
    });
    // Stop dragging; a click near the object shows its photo
    cv.addEventListener("pointerup", function (e) {
      // Short click (no drag)
      var o = view3d.objScreen, rect = cv.getBoundingClientRect();
      // Near the object marker
      if (drag && Math.abs(e.clientX - drag[0]) < 4 && o && Math.hypot(e.clientX - rect.left - o[0], e.clientY - rect.top - o[1]) < 20) $("mv-found").scrollIntoView();
      // End
      drag = null;
    });
    // Wheel zoom
    cv.addEventListener("wheel", function (e) { e.preventDefault(); dist = Math.max(0.5, Math.min(30, dist * (e.deltaY > 0 ? 1.1 : 0.9))); draw(); }, { passive: false });
    // Two-finger pinch zoom
    cv.addEventListener("touchmove", function (e) {
      // Two fingers only
      if (e.touches.length !== 2) { pinch = null; return; }
      // Finger distance
      var d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
      // Zoom relative to the last distance
      if (pinch) { dist = Math.max(0.5, Math.min(30, dist * pinch / d)); draw(); }
      // Remember
      pinch = d;
    });
    // Redraw on resize
    window.addEventListener("resize", draw);
    // First view
    draw();
  }
})();
