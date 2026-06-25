// ============================================================
// PokéDreamer Docs — main.js
// Navigation, scroll effects, chart rendering, interactions
// ============================================================

document.addEventListener('DOMContentLoaded', () => {

  // ── Navbar scroll effect ────────────────────────────────────
  const navbar = document.getElementById('navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      navbar.classList.toggle('scrolled', window.scrollY > 30);
    }, { passive: true });
  }

  // ── Mobile nav toggle ───────────────────────────────────────
  const toggle = document.getElementById('navToggle');
  const navLinks = document.getElementById('navLinks');
  if (toggle && navLinks) {
    toggle.addEventListener('click', () => {
      navLinks.classList.toggle('open');
    });
    // Close nav when link is clicked
    navLinks.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => navLinks.classList.remove('open'));
    });
  }

  // ── Intersection Observer for timeline animations ──────────
  const observerOptions = { threshold: 0.15 };
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, observerOptions);

  document.querySelectorAll('.timeline-item').forEach(el => observer.observe(el));

  // ── Active sidebar link tracking ────────────────────────────
  const sidebarLinks = document.querySelectorAll('.sidebar-link[href^="#"]');
  if (sidebarLinks.length > 0) {
    const headings = Array.from(
      document.querySelectorAll('.prose h2, .prose h3')
    ).filter(h => h.id);

    const headingObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          sidebarLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
          });
        }
      });
    }, { rootMargin: '-80px 0px -60% 0px' });

    headings.forEach(h => headingObserver.observe(h));
  }

  // ── Copy code buttons ───────────────────────────────────────
  window.copyCode = function(btn) {
    const pre = btn.closest('.code-block').querySelector('pre');
    if (!pre) return;
    navigator.clipboard.writeText(pre.textContent.trim()).then(() => {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = 'Copy';
        btn.classList.remove('copied');
      }, 2000);
    }).catch(() => {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = pre.textContent.trim();
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = 'Copy';
        btn.classList.remove('copied');
      }, 2000);
    });
  };

  // ── Theme toggle logic ──────────────────────────────────────
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
      const newTheme = currentTheme === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
      
      // Redraw charts since colors depend on the theme
      redrawCharts();
    });
  }

  // ── Chart initialization ────────────────────────────────────
  window.redrawCharts = function() {
    const driftCanvas = document.getElementById('driftChart');
    if (driftCanvas) {
      drawDriftChart(driftCanvas);
    }
    const rssmCanvas = document.getElementById('rssmChart');
    if (rssmCanvas) {
      drawRSSMChart(rssmCanvas);
    }
  };
  
  redrawCharts();

  // ── Smooth scroll for anchor links ─────────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
});

// ============================================================
// Chart Drawing Functions
// ============================================================

function drawDriftChart(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.parentElement.clientWidth || 800;
  const H = 300;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(dpr, dpr);

  const steps = [1, 5, 10, 15, 20, 25, 29];
  const ss    = [3.72, 3.32, 3.30, 3.33, 3.36, 3.42, 3.47];
  const tf    = [4.06, 5.08, 6.47, 7.81, 9.13, 9.72, 10.44];

  const pad = { top: 24, right: 40, bottom: 48, left: 56 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const maxVal = 11.5;

  const xScale = i => pad.left + (i / (steps.length - 1)) * chartW;
  const yScale = v => pad.top + chartH - (v / maxVal) * chartH;

  // Background
  ctx.fillStyle = 'transparent';
  ctx.fillRect(0, 0, W, H);

  // Theme detection
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const gridColor = isLight ? 'rgba(15, 23, 42, 0.06)' : 'rgba(255,255,255,0.05)';
  const subGridColor = isLight ? 'rgba(15, 23, 42, 0.04)' : 'rgba(255,255,255,0.04)';
  const labelColor = isLight ? 'rgba(71, 85, 105, 0.8)' : 'rgba(148,163,184,0.6)';
  const tfAreaColor = isLight ? 'rgba(245,158,11,0.04)' : 'rgba(245,158,11,0.08)';
  const ssAreaColor = isLight ? 'rgba(6,182,212,0.04)' : 'rgba(6,182,212,0.08)';

  // Grid lines
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let g = 0; g <= 5; g++) {
    const y = pad.top + (g / 5) * chartH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + chartW, y);
    ctx.stroke();
    const lbl = ((maxVal * (5 - g) / 5)).toFixed(1);
    ctx.fillStyle = labelColor;
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(lbl, pad.left - 8, y + 4);
  }

  // X axis labels
  ctx.textAlign = 'center';
  steps.forEach((s, i) => {
    const x = xScale(i);
    ctx.fillStyle = labelColor;
    ctx.font = '10px Inter';
    ctx.fillText(`Step ${s}`, x, H - 8);
    ctx.strokeStyle = subGridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + chartH);
    ctx.stroke();
  });

  // Axis labels
  ctx.save();
  ctx.translate(14, pad.top + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = labelColor;
  ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Tile Error (Manhattan)', 0, 0);
  ctx.restore();

  // TF area fill
  ctx.beginPath();
  tf.forEach((v, i) => i === 0 ? ctx.moveTo(xScale(i), yScale(v)) : ctx.lineTo(xScale(i), yScale(v)));
  ctx.lineTo(xScale(steps.length - 1), yScale(0));
  ctx.lineTo(xScale(0), yScale(0));
  ctx.closePath();
  ctx.fillStyle = tfAreaColor;
  ctx.fill();

  // SS area fill
  ctx.beginPath();
  ss.forEach((v, i) => i === 0 ? ctx.moveTo(xScale(i), yScale(v)) : ctx.lineTo(xScale(i), yScale(v)));
  ctx.lineTo(xScale(steps.length - 1), yScale(0));
  ctx.lineTo(xScale(0), yScale(0));
  ctx.closePath();
  ctx.fillStyle = ssAreaColor;
  ctx.fill();

  // TF line
  drawLine(ctx, steps, tf, xScale, yScale, '#f59e0b', 2.5);

  // SS line
  drawLine(ctx, steps, ss, xScale, yScale, '#06b6d4', 2.5);

  // Data points
  tf.forEach((v, i) => drawDot(ctx, xScale(i), yScale(v), '#f59e0b', 4));
  ss.forEach((v, i) => drawDot(ctx, xScale(i), yScale(v), '#22d3ee', 4));

  // Legend
  const legendX = pad.left + chartW - 200;
  const legendY = pad.top + 12;
  const legendLabelColor = isLight ? 'rgba(15, 23, 42, 0.9)' : 'rgba(148,163,184,0.9)';
  drawLegend(ctx, legendX, legendY, '#22d3ee', 'Scheduled Sampling (SS)', legendLabelColor);
  drawLegend(ctx, legendX, legendY + 22, '#f59e0b', 'Teacher Forcing (TF)', legendLabelColor);
}

function drawLine(ctx, steps, data, xScale, yScale, color, width) {
  ctx.beginPath();
  data.forEach((v, i) => i === 0 ? ctx.moveTo(xScale(i), yScale(v)) : ctx.lineTo(xScale(i), yScale(v)));
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.stroke();
}

function drawDot(ctx, x, y, color, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.5)';
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawLegend(ctx, x, y, color, label, textColor) {
  ctx.fillStyle = color;
  ctx.fillRect(x, y, 16, 3);
  ctx.fillStyle = textColor || 'rgba(148,163,184,0.9)';
  ctx.font = '11px Inter';
  ctx.textAlign = 'left';
  ctx.fillText(label, x + 22, y + 6);
}

function drawRSSMChart(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.parentElement.clientWidth || 800;
  const H = 280;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(dpr, dpr);

  const epochs = [1, 2, 3, 4];
  const trainRecon = [0.1379, 0.1144, 0.1068, 0.1015];
  const valRecon   = [0.1256, 0.1110, 0.1142, 0.1003];
  const trainKL    = [0.0078, 0.0063, 0.0422, 0.0005];

  const pad = { top: 24, right: 40, bottom: 48, left: 64 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const maxVal = 0.17;

  const xScale = i => pad.left + ((i) / (epochs.length - 1)) * chartW;
  const yScale = v => pad.top + chartH - (v / maxVal) * chartH;

  // Theme detection
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const gridColor = isLight ? 'rgba(15, 23, 42, 0.06)' : 'rgba(255,255,255,0.05)';
  const labelColor = isLight ? 'rgba(71, 85, 105, 0.8)' : 'rgba(148,163,184,0.6)';
  const legendLabelColor = isLight ? 'rgba(15, 23, 42, 0.9)' : 'rgba(148,163,184,0.9)';

  // Grid
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let g = 0; g <= 5; g++) {
    const y = pad.top + (g / 5) * chartH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + chartW, y);
    ctx.stroke();
    const lbl = (maxVal * (5 - g) / 5).toFixed(3);
    ctx.fillStyle = labelColor;
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(lbl, pad.left - 8, y + 4);
  }

  // X labels
  epochs.forEach((e, i) => {
    ctx.fillStyle = labelColor;
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(`Epoch ${e}`, xScale(i), H - 10);
  });

  // Y label
  ctx.save();
  ctx.translate(14, pad.top + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = labelColor;
  ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Reconstruction Loss', 0, 0);
  ctx.restore();

  // Lines
  drawLine(ctx, epochs, trainRecon, xScale, yScale, '#9d5ff5', 2.5);
  drawLine(ctx, epochs, valRecon, xScale, yScale, '#22d3ee', 2.5);

  // Dots
  trainRecon.forEach((v, i) => drawDot(ctx, xScale(i), yScale(v), '#9d5ff5', 5));
  valRecon.forEach((v, i) => drawDot(ctx, xScale(i), yScale(v), '#22d3ee', 5));

  // Best model star marker on epoch 4 val
  const bx = xScale(3), by = yScale(0.1003);
  ctx.beginPath();
  ctx.arc(bx, by, 8, 0, Math.PI * 2);
  ctx.strokeStyle = '#fbbf24';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = isLight ? '#b45309' : '#fbbf24';
  ctx.font = 'bold 10px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Best ★', bx + 12, by + 4);

  // Legend
  const lx = pad.left + 10, ly = pad.top + 10;
  drawLegend(ctx, lx, ly, '#9d5ff5', 'Train Recon', legendLabelColor);
  drawLegend(ctx, lx, ly + 22, '#22d3ee', 'Val Recon', legendLabelColor);
}
