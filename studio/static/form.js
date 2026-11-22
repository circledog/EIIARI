// Parameter form built from a task's schema (same schema the server validates with).
// Types: float, int (slider + number), bool (checkbox), choice (select), string (text).

// Check one value against its declaration; returns an error text or ""
function checkParam(p, v) {
  // Numbers: finite, whole for int, inside the range
  if (p.type === "float" || p.type === "int") {
    // Not a number
    if (typeof v !== "number" || !isFinite(v)) return "必須是數字";
    // Fraction for an int
    if (p.type === "int" && Math.round(v) !== v) return "必須是整數";
    // Out of range
    if (v < p.min || v > p.max) return "必須介於 " + p.min + " 和 " + p.max + " 之間";
  }
  // Text length
  if (p.type === "string" && v.length > (p.max_length || 200)) return "太長";
  // Valid
  return "";
}

// Build the form; onChange(name, value, valid) is called on every edit
function buildForm(container, schema, onChange) {
  // name -> {p, get, set, row, err}
  var fields = {};
  // Group name -> fieldset
  var groups = {};
  // One row per parameter
  schema.forEach(function (p) {
    // Fieldset of the parameter's group
    var g = p.group || "";
    // Create the fieldset the first time
    if (!groups[g]) {
      // New fieldset
      groups[g] = document.createElement("fieldset");
      // Legend with the translated group name
      if (g) groups[g].innerHTML = "<legend>" + t(g) + "</legend>";
      // Add it to the form
      container.appendChild(groups[g]);
    }
    // Row element
    var row = document.createElement("div");
    // Grid row
    row.className = "param";
    // Label (live parameters marked)
    var label = document.createElement("label");
    // Translated label
    label.textContent = (p.live ? "⚡ " : "") + t(p.label || p.name);
    // Into the row
    row.appendChild(label);
    // get() / set() for this parameter
    var get, set;
    // Numbers: slider and number box kept in sync
    if (p.type === "float" || p.type === "int") {
      // Slider
      var range = document.createElement("input");
      // Number box
      var num = document.createElement("input");
      // Both inputs share the limits
      [range, num].forEach(function (el, i) {
        // Input type
        el.type = i === 0 ? "range" : "number";
        // Limits and step
        el.min = p.min; el.max = p.max; el.step = p.step || (p.type === "int" ? 1 : "any");
      });
      // Into the row
      row.appendChild(range); row.appendChild(num);
      // Read the number box
      get = function () { return num.value === "" ? NaN : Number(num.value); };
      // Write both
      set = function (v) { range.value = v; num.value = v; };
      // Slider moved: copy to the box
      range.addEventListener("input", function () { num.value = range.value; changed(); });
      // Box edited: copy to the slider
      num.addEventListener("input", function () { range.value = num.value; changed(); });
    } else if (p.type === "bool") {
      // Checkbox
      var cb = document.createElement("input");
      // Type
      cb.type = "checkbox";
      // Into the row
      row.appendChild(cb);
      // Spacer for the grid
      row.appendChild(document.createElement("span"));
      // Read it
      get = function () { return cb.checked; };
      // Write it
      set = function (v) { cb.checked = !!v; };
      // Changed
      cb.addEventListener("change", changed);
    } else if (p.type === "choice") {
      // Select box
      var sel = document.createElement("select");
      // One option per choice
      p.choices.forEach(function (c) {
        // Option element
        var o = document.createElement("option");
        // Value as JSON so numbers stay numbers
        o.value = JSON.stringify(c[0]);
        // Translated label
        o.textContent = t(c[1]);
        // Add it
        sel.appendChild(o);
      });
      // Into the row
      row.appendChild(sel);
      // Spacer for the grid
      row.appendChild(document.createElement("span"));
      // Read the decoded value
      get = function () { return JSON.parse(sel.value); };
      // Select the matching option
      set = function (v) { sel.value = JSON.stringify(v); };
      // Changed
      sel.addEventListener("change", changed);
    } else {
      // Text box
      var tx = document.createElement("input");
      // Length limit
      tx.maxLength = p.max_length || 200;
      // Into the row
      row.appendChild(tx);
      // Spacer for the grid
      row.appendChild(document.createElement("span"));
      // Read it
      get = function () { return tx.value; };
      // Write it
      set = function (v) { tx.value = v; };
      // Changed
      tx.addEventListener("input", changed);
    }
    // Help text
    if (p.help) {
      // Small text under the row
      var h = document.createElement("div");
      // Class for styling
      h.className = "help";
      // Translated help
      h.textContent = t(p.help);
      // Into the row
      row.appendChild(h);
    }
    // Error line
    var err = document.createElement("div");
    // Class for styling
    err.className = "perr";
    // Into the row
    row.appendChild(err);
    // Into the fieldset
    groups[g].appendChild(row);
    // Validate and report an edit
    function changed() {
      // Current value and its error
      var v = get(), e = checkParam(p, v);
      // Show the error
      showError(p.name, e);
      // Tell the page
      if (onChange) onChange(p.name, v, !e);
    }
    // Start with the default
    set(p.default);
    // Remember the field
    fields[p.name] = { p: p, get: get, set: set, row: row, err: err };
  });
  // Show / clear an error for a field
  function showError(name, text) {
    // The field
    var f = fields[name];
    // Unknown field
    if (!f) return;
    // Translated error text
    f.err.textContent = text ? t(text) : "";
    // Red outline
    f.row.classList.toggle("invalid", !!text);
  }
  // Public interface
  return {
    // Every value, and whether all are valid
    values: function () {
      // name -> value
      var out = {}, ok = true;
      // Every field
      Object.keys(fields).forEach(function (k) {
        // Current value
        out[k] = fields[k].get();
        // Validity
        var e = checkParam(fields[k].p, out[k]);
        // Show the error
        showError(k, e);
        // Any error makes the form invalid
        if (e) ok = false;
      });
      // Values and validity
      return { values: out, valid: ok };
    },
    // Set several values (unknown names ignored)
    setValues: function (vals) {
      // Every given value
      Object.keys(vals || {}).forEach(function (k) {
        // Known field: set and clear its error
        if (fields[k]) { fields[k].set(vals[k]); showError(k, ""); }
      });
    },
    // Server-side errors {name: message}
    setErrors: function (errs) {
      // Clear all, then show the given ones
      Object.keys(fields).forEach(function (k) { showError(k, (errs || {})[k] || ""); });
    },
    // While running only live parameters can be edited
    setRunning: function (running) {
      // Every field
      Object.keys(fields).forEach(function (k) {
        // Disable fixed parameters while running
        fields[k].row.querySelectorAll("input, select").forEach(function (el) {
          // Disabled state
          el.disabled = running && !fields[k].p.live;
        });
      });
    },
    // Whether a parameter can change while running
    isLive: function (name) { return !!(fields[name] && fields[name].p.live); }
  };
}
