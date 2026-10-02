// The gate: the AI needs a laptop or desktop computer whose browser can reach its graphics chip (WebGPU). Anything
// else (a phone, a tablet, a phone asking for the "desktop site", a TV, a computer whose browser has no WebGPU or no
// working graphics chip) gets one screen instead of the website: what went wrong, in a few words, and where to go.
//
// A classic script in <head>, before anything draws. Phones and tablets are told apart from the browser's own report
// and from the hardware (a touch screen with no mouse, an ARM phone processor, a phone's graphics chip), so "Request
// desktop site" doesn't slip past. Windows PCs and Chromebooks always pass, whatever their shape: they run the AI.
// Plain, older JavaScript, so even an old phone can show the message. ?gate=phone (or tablet, other, browser, nogpu,
// software) previews a screen on any device; nothing turns the gate off.

(function () {
  "use strict";
  var html = document.documentElement;
  var nav = navigator;
  var ua = nav.userAgent || "";
  var touch = nav.maxTouchPoints || 0;
  var uad = nav.userAgentData;
  var plat = nav.platform || "";
  var PC = /Windows NT|CrOS/;

  function mq(q) {
    try {
      return matchMedia(q).matches;
    } catch (e) {
      return false;
    }
  }

  function screenKind() {
    var s = Math.min(screen.width || 0, screen.height || 0);
    return s && s < 600 ? "phone" : "tablet";
  }

  /** "phone", "tablet", "other" (TV, console, headset, watch) or "" (a computer). */
  function kindOf() {
    if (/Xbox|PlayStation|Nintendo|SMART-TV|SmartTV|HbbTV|NetCast|Web0S|webOS|Tizen|CrKey|BRAVIA|AppleTV|GoogleTV|Android TV|AFT[A-Z]|OculusBrowser|Quest \d|Pico|Wear OS|KAIOS/.test(ua))
      return "other";
    if (/Windows Phone|IEMobile/.test(ua)) return "phone";
    if (PC.test(ua)) return "";
    if (/iPhone|iPod/.test(ua)) return "phone";
    if (/iPad/.test(ua) || (/Macintosh/.test(ua) && touch > 1)) return screenKind(); // in desktop mode iPads (and iPhones) say "Macintosh"
    if (/Android/.test(ua)) return /Mobile/.test(ua) ? "phone" : "tablet";
    if (/Mobi|Opera Mini|BlackBerry|BB10/i.test(ua)) return "phone";
    if (/Tablet|Silk|Kindle|PlayBook/.test(ua)) return "tablet";
    if (uad && uad.mobile) return "phone";
    if (uad && /Android|iOS/i.test(uad.platform || "")) return screenKind();
    // "Request desktop site": the browser claims to be a computer; the hardware says otherwise
    if (/^(iPhone|iPod)/.test(plat)) return "phone";
    if (/^iPad/.test(plat)) return "tablet";
    if (/Macintosh/.test(ua)) return ""; // a Mac: Macs have no touch screen
    if (touch > 0 && /Linux (arm|aarch64)/i.test(plat)) return screenKind();
    if (touch > 0 && mq("(pointer: coarse)") && mq("(hover: none)") && !mq("(any-pointer: fine)")) return screenKind();
    return "";
  }

  /** What the graphics chip WebGPU offers says: "" (fine), "nogpu", "software", or "phone"/"tablet". */
  function judge(a) {
    if (!a) return "nogpu";
    var info = a.info || {};
    if (a.isFallbackAdapter || info.isFallbackAdapter) return "software";
    var said = [info.vendor, info.architecture, info.device, info.description].join(" ");
    if (/swiftshader|llvmpipe|lavapipe|softpipe|microsoft basic render|\bwarp\b/i.test(said)) return "software";
    // a phone or tablet in desktop mode that slipped through: its graphics chip is a phone's
    if (touch > 0 && !PC.test(ua) && !/Macintosh/.test(ua) && /^(qualcomm|arm|imagination|img-tec|samsung|huawei|mediatek)$/i.test(info.vendor || ""))
      return screenKind();
    return "";
  }

  var gate = {
    gated: false,
    kind: "",
    ready: null, // resolves true when the website may run
    adapter: null, // a promise of the graphics chip WebGPU offers, shared with the AI's engine
  };
  window.HDGate = gate;

  function close(kind) {
    if (gate.gated) return;
    gate.gated = true;
    gate.kind = kind;
    html.classList.add("gated");
    html.setAttribute("data-gate", kind);
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
    else build();
    try {
      window.dispatchEvent(new CustomEvent("hd-gated", { detail: kind }));
    } catch (e) {}
  }

  // ================================================================== the words
  function browserName() {
    var brands = ((uad && uad.brands) || []).map(function (b) {
      return b.brand;
    });
    if (brands.indexOf("Microsoft Edge") >= 0 || /Edg\//.test(ua)) return "Edge";
    if (brands.indexOf("Brave") >= 0) return "Brave";
    if (brands.indexOf("Opera") >= 0 || /OPR\//.test(ua)) return "Opera";
    if (/Firefox\//.test(ua)) return "Firefox";
    if (/SamsungBrowser/.test(ua)) return "Samsung Internet";
    if (brands.indexOf("Google Chrome") >= 0 || /Chrome\/|CriOS/.test(ua)) return "Chrome";
    if (/Safari\//.test(ua)) return "Safari";
    return "";
  }

  function osName() {
    return /Windows/.test(ua) ? "windows" : /Macintosh/.test(ua) ? "mac" : /CrOS/.test(ua) ? "chromeos" : /Linux/.test(ua) ? "linux" : "";
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return "&#" + c.charCodeAt(0) + ";";
    });
  }

  function extend(a, b) {
    var o = {};
    var k;
    for (k in a) o[k] = a[k];
    for (k in b) o[k] = b[k];
    return o;
  }

  function words(k) {
    var b = browserName();
    var os = osName();
    var plan = "This website downloads a 500 MB AI and makes a graphics chip invent faces out of pure static, several a second.";
    var device = {
      kicker: "Error 404 · laptop not found",
      stamp: "Needs a laptop",
      ask: "Open it on a laptop or a desktop computer, in Chrome, Edge or Safari. Everything works there.",
      share: true,
    };
    if (k === "phone")
      return extend(device, {
        where: "…on phones.",
        body: plan + " We showed your phone the plan. It got warm, dimmed its screen and asked to lie down.",
        note: "Switching to “Desktop site” won’t fool it. We checked.",
      });
    if (k === "tablet")
      return extend(device, {
        where: "…on tablets.",
        body: plan + " Your tablet read the plan, quietly closed every other app, and is now pretending to be asleep.",
        note: "Switching to “Desktop site” won’t fool it. We checked.",
      });
    if (k === "other")
      return extend(device, {
        where: "…on whatever this is.",
        body: plan + " Whatever this is (a TV? a console? a very smart fridge?), it isn’t the laptop the AI was promised.",
        note: "No judgement. It’s a lovely screen.",
      });
    if (k === "browser")
      return {
        kicker: "Error · no WebGPU",
        stamp: "Needs WebGPU",
        where: "…in this browser.",
        body:
          "The AI draws faces on your computer’s graphics chip through WebGPU, and " +
          (b || "this browser") +
          " doesn’t offer WebGPU here. No graphics chip, no faces. Just these escapees.",
        ask:
          os === "mac"
            ? b === "Safari"
              ? Number((/Version\/(\d+)/.exec(ua) || [])[1]) >= 26
                ? "Open this page in Chrome or Edge, or update macOS and Safari to their newest versions (Apple menu › System Settings › General › Software Update)."
                : "Update Safari to version 26 or later (Apple menu › System Settings › General › Software Update), or open this page in Chrome or Edge."
              : "Open this page in Chrome, Edge, or Safari 26 or later."
            : "Open this page in Chrome or Edge" + (b === "Firefox" ? ", or update Firefox to its newest version." : "."),
        note: "It’s not you. It’s " + (b || "the browser") + ".",
      };
    // nogpu / software: WebGPU is there, but no real graphics chip answered
    var name = b || "your browser";
    var setting =
      b === "Firefox"
        ? "Turn on hardware acceleration: Firefox › Settings › General › Performance › untick “Use recommended performance settings” › tick “Use hardware acceleration when available”"
        : b === "Safari" || !b
          ? ""
          : "Turn on graphics acceleration: " + b + " › Settings › System" + (b === "Edge" ? " and performance" : "") + " › “Use graphics acceleration when available”";
    var linux =
      os === "linux" && b !== "Firefox"
        ? ' On Linux, also turn on <span class="gate-flag">chrome://flags/#enable-unsafe-webgpu</span> and <span class="gate-flag">chrome://flags/#enable-vulkan</span>.'
        : "";
    return {
      kicker: k === "software" ? "Error · graphics acceleration off" : "Error · graphics chip not answering",
      stamp: "Needs a graphics chip",
      where: "…without a graphics chip.",
      body:
        k === "software"
          ? "Your browser is faking a graphics chip with the processor, and the AI can tell. It refuses to draw like this."
          : "Your browser has WebGPU, but when the AI knocked, no graphics chip answered. Usually graphics acceleration is switched off, the graphics driver is old, or this is a virtual machine or a remote desktop with no chip at all.",
      ask:
        (setting ? esc(setting) + ", then restart " + esc(name) + "." : "") +
        linux +
        " Still nothing? Update your graphics driver, or open this page on another computer.",
      html: true,
      note: "It’s not you. It’s the graphics driver. Probably.",
    };
  }

  // ================================================================== the screen
  var built = false;
  function build() {
    if (built || !document.body) return;
    built = true;
    var w = words(gate.kind);
    document.title = /phone|tablet|other/.test(gate.kind) ? "HumanDiT: this needs a laptop" : "HumanDiT: this needs a graphics chip";
    var title = ["This website", "does not exist."];
    // each letter can lose its footing; each word stays in one piece
    var line = function (text) {
      return (
        '<span class="gate-line">' +
        text
          .split(" ")
          .map(function (word) {
            return '<span class="gw">' + word.split("").map(function (c) {
              return '<span class="gl">' + esc(c) + "</span>";
            }).join("") + "</span>";
          })
          .join(" ") +
        "</span>"
      );
    };
    var root = document.createElement("main");
    root.className = "gate";
    root.setAttribute("aria-labelledby", "gate-title");
    root.innerHTML =
      '<a class="wordmark gate-mark" href="/"><span class="wordmark-dot" aria-hidden="true"></span>HumanDiT</a>' +
      '<div class="gate-zone" aria-hidden="true">' +
      '<canvas class="gate-static" width="96" height="64"></canvas>' +
      '<div class="gate-faces"></div>' +
      '<span class="gate-stamp">' + esc(w.stamp) + "</span>" +
      '<span class="gate-count">9 people who don’t exist, on the loose.</span>' +
      "</div>" +
      '<div class="gate-copy">' +
      '<p class="gate-kicker">' + esc(w.kicker) + "</p>" +
      '<h1 class="gate-title" id="gate-title"><span class="sr">' + title.join(" ") + '</span><span aria-hidden="true">' + line(title[0]) + " " + line(title[1]) + "</span></h1>" +
      '<p class="gate-where">' + esc(w.where) + "</p>" +
      '<p class="gate-body">' + esc(w.body) + "</p>" +
      '<p class="gate-ask">' + (w.html ? w.ask : esc(w.ask)) + "</p>" +
      '<div class="gate-actions">' +
      '<button class="btn btn-primary gate-copy-link" type="button"><span>Copy the link</span></button>' +
      (w.share && nav.share ? '<button class="btn btn-ghost gate-share" type="button"><span>Send it to my laptop</span></button>' : "") +
      "</div>" +
      '<p class="gate-note">' + esc(w.note) + "</p>" +
      "</div>";
    document.body.appendChild(root);

    var link = location.origin + "/";
    var copyBtn = root.querySelector(".gate-copy-link span");
    root.querySelector(".gate-copy-link").addEventListener("click", function () {
      var said = function (t) {
        copyBtn.textContent = t;
      };
      if (nav.clipboard && nav.clipboard.writeText)
        nav.clipboard.writeText(link).then(
          function () {
            said("Copied: " + location.host);
          },
          function () {
            said(location.host);
          },
        );
      else said(location.host);
    });
    var shareBtn = root.querySelector(".gate-share");
    if (shareBtn)
      shareBtn.addEventListener("click", function () {
        nav.share({ title: "HumanDiT", text: "Faces of people who don’t exist, made by an AI on your own graphics chip. Open it on a laptop:", url: link }).then(null, function () {});
      });
    chaos(root);
  }

  // ================================================================== the chaos
  // Nine of the AI's faces bounce around a box of static, spin, collide, and now and then glitch back into their own
  // drafts; a tap flings one. The headline's letters lose their footing every few seconds; a stamp slams down.
  // With reduced motion: the faces sit still and nothing shakes.
  function chaos(root) {
    var still = mq("(prefers-reduced-motion: reduce)");
    var zone = root.querySelector(".gate-zone");
    var pen = root.querySelector(".gate-faces");
    var count = root.querySelector(".gate-count");
    var stamp = root.querySelector(".gate-stamp");
    var sctx = root.querySelector(".gate-static").getContext("2d");
    var letters = [].slice.call(root.querySelectorAll(".gl"));
    var N = 9; // rows of assets/gate.jpg (scripts/site/build_gate.py)
    var DRAFTS = 4; // its columns: the finished face, then earlier drafts
    var SRC = 144;
    var faces = [];
    var W = 0;
    var H = 0;
    var S = 0;
    var flicked = 0;
    var last = 0;

    var noise = [];
    for (var n = 0; n < 4; n++) {
      var c = document.createElement("canvas");
      c.width = c.height = 64;
      var x = c.getContext("2d");
      var im = x.createImageData(64, 64);
      for (var p = 0; p < im.data.length; p += 4) {
        var v = (Math.random() * 255) | 0;
        im.data[p] = im.data[p + 1] = im.data[p + 2] = v;
        im.data[p + 3] = 255;
      }
      x.putImageData(im, 0, 0);
      noise.push(c);
    }
    var pick = function () {
      return noise[(Math.random() * noise.length) | 0];
    };
    var drawStatic = function () {
      var t = pick();
      var ox = (Math.random() * 64) | 0;
      var oy = (Math.random() * 64) | 0;
      for (var i = -1; i < 2; i++) for (var j = -1; j < 2; j++) sctx.drawImage(t, ox + i * 64, oy + j * 64);
    };
    drawStatic();

    var measure = function () {
      var r = zone.getBoundingClientRect();
      W = r.width;
      H = r.height;
      S = Math.round(Math.max(56, Math.min(124, Math.sqrt((W * H) / 24)))); // nine faces fill about a third of the box
      faces.forEach(function (f) {
        f.el.style.width = f.el.style.height = S + "px";
        f.x = Math.min(Math.max(0, f.x), Math.max(0, W - S));
        f.y = Math.min(Math.max(0, f.y), Math.max(0, H - S));
        place(f);
      });
    };

    var paint = function (f, glitch) {
      var g = f.ctx;
      var z = f.el.width;
      var d = glitch ? 1 + ((Math.random() * (DRAFTS - 1)) | 0) : 0;
      g.globalAlpha = 1;
      g.drawImage(sprite, d * SRC, f.row * SRC, SRC, SRC, 0, 0, z, z);
      if (!glitch) return;
      for (var s = 0; s < 3; s++) {
        var y = Math.random() * z;
        var h = 4 + Math.random() * z * 0.18;
        g.drawImage(f.el, 0, y, z, h, (Math.random() - 0.5) * z * 0.22, y, z, h);
      }
      g.globalAlpha = 0.25 + Math.random() * 0.55;
      g.drawImage(pick(), 0, 0, z, z);
      g.globalAlpha = 1;
    };

    var place = function (f) {
      f.el.style.transform = "translate3d(" + f.x.toFixed(1) + "px," + f.y.toFixed(1) + "px,0) rotate(" + f.r.toFixed(1) + "deg)";
    };

    var shake = function (el) {
      if (still) return;
      el.classList.remove("shake");
      void el.offsetWidth;
      el.classList.add("shake");
    };

    var flick = function (f) {
      var a = Math.random() * Math.PI * 2;
      var sp = 380 + Math.random() * 260;
      f.vx = Math.cos(a) * sp;
      f.vy = Math.sin(a) * sp;
      f.vr = (Math.random() - 0.5) * 900;
      f.glitchUntil = last + 380;
      flicked++;
      count.textContent =
        "You flicked " + flicked + (flicked === 1 ? " person who doesn’t exist." : " people who don’t exist.") + (flicked > 12 ? " They’re fine. Probably." : "");
      shake(zone);
    };

    var start = function () {
      measure();
      for (var i = 0; i < N; i++) {
        var cv = document.createElement("canvas");
        cv.width = cv.height = 112;
        cv.className = "gate-face";
        cv.style.width = cv.style.height = S + "px";
        pen.appendChild(cv);
        var a = Math.random() * Math.PI * 2;
        var sp = 70 + Math.random() * 90;
        var f = {
          el: cv,
          ctx: cv.getContext("2d"),
          row: i,
          x: Math.random() * Math.max(1, W - S),
          y: Math.random() * Math.max(1, H - S),
          vx: Math.cos(a) * sp,
          vy: Math.sin(a) * sp,
          r: (Math.random() - 0.5) * 40,
          vr: (Math.random() - 0.5) * 90,
          glitchUntil: 0,
          nextGlitch: 600 + Math.random() * 4000,
        };
        faces.push(f);
        paint(f, false);
        place(f);
        if (!still)
          cv.addEventListener(
            "pointerdown",
            (function (face) {
              return function (e) {
                e.preventDefault();
                flick(face);
              };
            })(f),
          );
      }
      if (still) {
        stamp.classList.add("in");
        return;
      }
      setTimeout(function () {
        stamp.classList.add("slam");
        shake(zone);
      }, 700);
      requestAnimationFrame(loop);
    };

    var lastStatic = 0;
    var nextBurst = 900;
    var burstUntil = 0;
    var loop = function (now) {
      requestAnimationFrame(loop);
      var dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
      last = now;
      if (now - lastStatic > 80) {
        drawStatic();
        lastStatic = now;
      }
      var damp = Math.pow(0.6, dt);
      for (var i = 0; i < faces.length; i++) {
        var f = faces[i];
        f.x += f.vx * dt;
        f.y += f.vy * dt;
        f.r += f.vr * dt;
        if (f.vx * f.vx + f.vy * f.vy > 160 * 160) {
          f.vx *= damp; // a flung face slows back down to a drift
          f.vy *= damp;
        }
        if (Math.abs(f.vr) > 60) f.vr *= damp;
        var kick = (Math.random() - 0.5) * 120;
        if (f.x < 0) (f.x = 0), (f.vx = Math.abs(f.vx)), (f.vr += kick);
        if (f.y < 0) (f.y = 0), (f.vy = Math.abs(f.vy)), (f.vr += kick);
        if (f.x > W - S) (f.x = W - S), (f.vx = -Math.abs(f.vx)), (f.vr += kick);
        if (f.y > H - S) (f.y = H - S), (f.vy = -Math.abs(f.vy)), (f.vr += kick);
        for (var j = i + 1; j < faces.length; j++) {
          var o = faces[j];
          var dx = o.x - f.x;
          var dy = o.y - f.y;
          var d2 = dx * dx + dy * dy;
          var min = S * 0.92;
          if (d2 > 0 && d2 < min * min) {
            var dd = Math.sqrt(d2);
            var nx = dx / dd;
            var ny = dy / dd;
            var push = (min - dd) / 2;
            f.x -= nx * push;
            f.y -= ny * push;
            o.x += nx * push;
            o.y += ny * push;
            var rel = (o.vx - f.vx) * nx + (o.vy - f.vy) * ny;
            if (rel < 0) {
              f.vx += rel * nx;
              f.vy += rel * ny;
              o.vx -= rel * nx;
              o.vy -= rel * ny;
            }
          }
        }
        if (now > f.nextGlitch) {
          f.glitchUntil = now + 140 + Math.random() * 260;
          f.nextGlitch = now + 1800 + Math.random() * 5200;
        }
        if (now < f.glitchUntil) paint(f, true);
        else if (f.glitchUntil) {
          f.glitchUntil = 0;
          paint(f, false);
        }
        place(f);
      }
      // the headline: every few seconds its letters lose their footing for half a second
      if (now > nextBurst) {
        burstUntil = now + 520;
        nextBurst = now + 2400 + Math.random() * 2600;
      }
      if (now < burstUntil) {
        if (((now / 45) | 0) % 2)
          letters.forEach(function (l) {
            var hit = Math.random() < 0.28;
            l.style.transform = hit
              ? "translate(" + ((Math.random() - 0.5) * 7).toFixed(1) + "px," + ((Math.random() - 0.5) * 9).toFixed(1) + "px) rotate(" + ((Math.random() - 0.5) * 18).toFixed(0) + "deg)"
              : "";
            l.className = hit && Math.random() < 0.4 ? "gl hot" : "gl";
          });
      } else if (burstUntil) {
        burstUntil = 0;
        letters.forEach(function (l) {
          l.style.transform = "";
          l.className = "gl";
        });
      }
    };

    stamp.addEventListener("pointerdown", function () {
      if (still) return;
      stamp.classList.remove("slam");
      void stamp.offsetWidth;
      stamp.classList.add("slam");
      shake(zone);
    });
    if (window.ResizeObserver) new ResizeObserver(measure).observe(zone);
    else window.addEventListener("resize", measure);

    var sprite = new Image();
    sprite.onload = start;
    sprite.onerror = function () {
      stamp.classList.add("in"); // no faces: the words still say it all
    };
    sprite.src = "/assets/gate.jpg";
  }

  // ================================================================== the decision
  var preview = (/[?&]gate=(phone|tablet|other|browser|nogpu|software)\b/.exec(location.search) || [])[1] || "";
  var kind = preview || kindOf();
  if (!kind && !("gpu" in nav)) kind = "browser";
  if (kind) {
    close(kind);
    gate.ready = Promise.resolve(false);
  } else {
    // a computer with WebGPU: ask for its fastest graphics chip; check one answers, and that it's a real one
    try {
      gate.adapter = nav.gpu.requestAdapter({ powerPreference: "high-performance" }).then(null, function () {
        return null;
      });
    } catch (e) {
      gate.adapter = Promise.resolve(null);
    }
    var verdict = gate.adapter.then(judge);
    verdict.then(function (k) {
      if (k) close(k);
    });
    // the page waits for the answer (usually a few hundredths of a second), never long: an answer that comes late
    // and is bad still closes the page when it arrives
    gate.ready = Promise.race([
      verdict.then(function (k) {
        return !k;
      }),
      new Promise(function (r) {
        setTimeout(function () {
          r(true);
        }, 2500);
      }),
    ]);
  }
})();
