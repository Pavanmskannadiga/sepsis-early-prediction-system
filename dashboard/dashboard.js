/* ========================================
   SEPSIS PREDICTION DASHBOARD — JS
   ======================================== */
(function () {
    'use strict';

    /* ---- Helpers ---- */
    const $ = (s, p = document) => p.querySelector(s);
    const $$ = (s, p = document) => [...p.querySelectorAll(s)];
    const lerp = (a, b, t) => a + (b - a) * t;
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    const dpr = window.devicePixelRatio || 1;

    function setupHiDPI(canvas) {
        const rect = canvas.getBoundingClientRect();
        const w = rect.width || canvas.width;
        const h = rect.height || canvas.height;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return { ctx, w, h };
    }

    /* ===================== NAVBAR ===================== */
    const navbar = $('.navbar');
    const hamburger = $('#navHamburger');
    const mobileNav = $('#mobileNav');

    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 40);
    });
    hamburger?.addEventListener('click', () => {
        mobileNav.classList.toggle('open');
    });
    $$('.mobile-nav a').forEach(a => a.addEventListener('click', () => mobileNav.classList.remove('open')));

    /* ===================== INTERSECTION OBSERVER ===================== */
    const animatedElements = new Set();
    const chartDrawn = {};

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                // Stagger support
                const delay = el.dataset.delay || 0;
                setTimeout(() => el.classList.add('visible'), +delay);
                animatedElements.add(el);

                // Trigger stat bar fills
                $$('.stat-bar-fill', el).forEach(bar => {
                    setTimeout(() => { bar.style.width = bar.dataset.width + '%'; }, +delay + 200);
                });

                // Trigger number counters
                $$('.stat-value[data-target]', el).forEach(sv => {
                    setTimeout(() => animateCounter(sv), +delay + 100);
                });

                // Canvas chart triggers
                if (el.querySelector('#rocCanvas') && !chartDrawn.roc) { chartDrawn.roc = true; setTimeout(drawROC, 300); }
                if (el.querySelector('#prCanvas') && !chartDrawn.pr) { chartDrawn.pr = true; setTimeout(drawPR, 300); }
                if (el.querySelector('#shapCanvas') && !chartDrawn.shap) { chartDrawn.shap = true; setTimeout(drawSHAP, 300); }
                if (el.querySelector('#crossHospitalCanvas') && !chartDrawn.cross) { chartDrawn.cross = true; setTimeout(drawCrossHospital, 300); }
                if (el.querySelector('#donutBefore') && !chartDrawn.donut) { chartDrawn.donut = true; setTimeout(drawDonuts, 300); }
                if (el.querySelector('#imbalanceCanvas') && !chartDrawn.imb) { chartDrawn.imb = true; setTimeout(drawImbalance, 300); }
            }
        });
    }, { threshold: 0.15 });

    $$('.animate-in').forEach(el => observer.observe(el));

    /* ===================== ANIMATED COUNTER ===================== */
    function animateCounter(el) {
        const target = parseFloat(el.dataset.target);
        const decimals = parseInt(el.dataset.decimals) || 0;
        const duration = 1800;
        const start = performance.now();
        function tick(now) {
            const t = clamp((now - start) / duration, 0, 1);
            const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
            const val = lerp(0, target, eased);
            el.textContent = val.toFixed(decimals);
            if (t < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    /* ===================== CHART UTILS ===================== */
    function chartColors() {
        return {
            grid: 'rgba(255,255,255,0.05)',
            axis: 'rgba(255,255,255,0.15)',
            text: 'rgba(255,255,255,0.45)',
            accent1: '#667eea',
            accent2: '#a78bfa',
            accent3: '#00d2ff',
            green: '#22c55e',
            red: '#ef4444',
            orange: '#fb923c',
            yellow: '#eab308',
            blue: '#38bdf8',
        };
    }

    function drawGridAndAxes(ctx, w, h, pad, xLabel, yLabel, c) {
        ctx.strokeStyle = c.grid;
        ctx.lineWidth = 0.5;
        const plotW = w - pad.l - pad.r;
        const plotH = h - pad.t - pad.b;
        for (let i = 0; i <= 5; i++) {
            const y = pad.t + (plotH / 5) * i;
            ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
            ctx.fillStyle = c.text; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'right';
            ctx.fillText((1 - i / 5).toFixed(1), pad.l - 8, y + 4);
        }
        for (let i = 0; i <= 5; i++) {
            const x = pad.l + (plotW / 5) * i;
            ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
            ctx.fillStyle = c.text; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText((i / 5).toFixed(1), x, h - pad.b + 18);
        }
        // Axes
        ctx.strokeStyle = c.axis; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
        // Labels
        ctx.fillStyle = c.text; ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(xLabel, pad.l + plotW / 2, h - 4);
        ctx.save();
        ctx.translate(14, pad.t + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();
    }

    /* Animate a curve drawing */
    function animateCurve(ctx, points, color, w, h, pad, duration, lineWidth, fillBelow) {
        const plotW = w - pad.l - pad.r;
        const plotH = h - pad.t - pad.b;
        const mapped = points.map(p => ({
            x: pad.l + p[0] * plotW,
            y: pad.t + (1 - p[1]) * plotH
        }));
        const start = performance.now();
        function draw(now) {
            const t = clamp((now - start) / duration, 0, 1);
            const eased = 1 - Math.pow(1 - t, 2);
            const n = Math.floor(eased * (mapped.length - 1)) + 1;

            // Clear only the plotting area to avoid grid flicker — we just re-overlay
            ctx.save();
            ctx.beginPath();
            ctx.rect(pad.l, pad.t, plotW, plotH);
            ctx.clip();

            if (fillBelow) {
                ctx.beginPath();
                ctx.moveTo(mapped[0].x, h - pad.b);
                for (let i = 0; i < n; i++) ctx.lineTo(mapped[i].x, mapped[i].y);
                ctx.lineTo(mapped[n - 1].x, h - pad.b);
                ctx.closePath();
                const grad = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
                grad.addColorStop(0, color.replace(')', ',0.2)').replace('rgb', 'rgba'));
                grad.addColorStop(1, 'transparent');
                ctx.fillStyle = grad;
                ctx.fill();
            }

            ctx.beginPath();
            ctx.moveTo(mapped[0].x, mapped[0].y);
            for (let i = 1; i < n; i++) ctx.lineTo(mapped[i].x, mapped[i].y);
            ctx.strokeStyle = color;
            ctx.lineWidth = lineWidth || 2.5;
            ctx.lineJoin = 'round';
            ctx.stroke();

            // Glow
            ctx.strokeStyle = color;
            ctx.lineWidth = (lineWidth || 2.5) + 4;
            ctx.globalAlpha = 0.15;
            ctx.beginPath();
            ctx.moveTo(mapped[0].x, mapped[0].y);
            for (let i = 1; i < n; i++) ctx.lineTo(mapped[i].x, mapped[i].y);
            ctx.stroke();
            ctx.globalAlpha = 1;

            ctx.restore();
            if (t < 1) requestAnimationFrame(draw);
        }
        requestAnimationFrame(draw);
    }

    /* ===================== ROC CURVE ===================== */
    function generateROCPoints(auc) {
        // Generate a realistic ROC curve using a power function tuned for given AUC
        // y = x^p where the AUC ≈ 1/(p+1), so p = 1/AUC - 1 (for the inverse)
        // Actually: for ROC, TPR = FPR^(1/c) where c controls curvature
        // AUC ≈ c/(c+1), so c = AUC/(1-AUC)
        const c = auc / (1 - auc);
        const pts = [];
        for (let i = 0; i <= 100; i++) {
            const fpr = i / 100;
            const tpr = Math.pow(fpr, 1 / c);
            pts.push([fpr, tpr]);
        }
        return pts;
    }

    function drawROC() {
        const canvas = $('#rocCanvas');
        if (!canvas) return;
        const { ctx, w, h } = setupHiDPI(canvas);
        const c = chartColors();
        const pad = { t: 30, r: 30, l: 50, b: 45 };

        drawGridAndAxes(ctx, w, h, pad, 'False Positive Rate', 'True Positive Rate', c);

        // Diagonal
        const plotW = w - pad.l - pad.r;
        const plotH = h - pad.t - pad.b;
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = 'rgba(255,255,255,0.12)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.l, h - pad.b);
        ctx.lineTo(pad.l + plotW, pad.t);
        ctx.stroke();
        ctx.setLineDash([]);

        // Curves
        const models = [
            { auc: 0.79, color: c.blue, lw: 1.5, label: 'LR (0.79)' },
            { auc: 0.84, color: c.green, lw: 1.5, label: 'RF (0.84)' },
            { auc: 0.86, color: c.yellow, lw: 1.5, label: 'LSTM (0.86)' },
            { auc: 0.88, color: c.orange, lw: 1.5, label: 'XGB (0.88)' },
            { auc: 0.91, color: c.accent1, lw: 3, label: 'Ours (0.91)' },
        ];

        models.forEach((m, idx) => {
            const pts = generateROCPoints(m.auc);
            setTimeout(() => animateCurve(ctx, pts, m.color, w, h, pad, 1200, m.lw, m.auc === 0.91), idx * 200);
        });

        // Legend
        setTimeout(() => {
            const lx = pad.l + plotW - 130;
            let ly = pad.t + 16;
            ctx.fillStyle = 'rgba(10,10,26,0.7)';
            ctx.roundRect?.(lx - 10, ly - 12, 140, models.length * 20 + 10, 8);
            ctx.fill?.();
            models.slice().reverse().forEach(m => {
                ctx.fillStyle = m.color;
                ctx.fillRect(lx, ly - 4, 14, 3);
                ctx.fillStyle = c.text;
                ctx.font = '10px Inter, sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(m.label, lx + 20, ly);
                ly += 20;
            });
        }, 1600);
    }

    /* ===================== PR CURVE ===================== */
    function generatePRPoints(aucpr) {
        // precision = aucpr / (aucpr + (1-aucpr)*recall)  — rough approximation
        const pts = [];
        for (let i = 0; i <= 100; i++) {
            const recall = i / 100;
            // Use a model: precision = 1 - (1-aucpr)*recall^k
            const k = 1.5;
            const precision = Math.max(0, 1 - (1 - aucpr) * Math.pow(recall, k));
            pts.push([recall, precision]);
        }
        return pts;
    }

    function drawPR() {
        const canvas = $('#prCanvas');
        if (!canvas) return;
        const { ctx, w, h } = setupHiDPI(canvas);
        const c = chartColors();
        const pad = { t: 30, r: 30, l: 50, b: 45 };

        drawGridAndAxes(ctx, w, h, pad, 'Recall', 'Precision', c);

        const models = [
            { auc: 0.58, color: c.blue, lw: 1.5, label: 'LR (0.58)' },
            { auc: 0.65, color: c.green, lw: 1.5, label: 'RF (0.65)' },
            { auc: 0.69, color: c.yellow, lw: 1.5, label: 'LSTM (0.69)' },
            { auc: 0.72, color: c.orange, lw: 1.5, label: 'XGB (0.72)' },
            { auc: 0.78, color: c.accent2, lw: 3, label: 'Ours (0.78)' },
        ];

        models.forEach((m, idx) => {
            const pts = generatePRPoints(m.auc);
            setTimeout(() => animateCurve(ctx, pts, m.color, w, h, pad, 1200, m.lw, m.auc === 0.78), idx * 200);
        });

        setTimeout(() => {
            const plotW = w - pad.l - pad.r;
            const lx = pad.l + plotW - 130;
            let ly = pad.t + 16;
            models.slice().reverse().forEach(m => {
                ctx.fillStyle = m.color;
                ctx.fillRect(lx, ly - 4, 14, 3);
                ctx.fillStyle = c.text;
                ctx.font = '10px Inter, sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(m.label, lx + 20, ly);
                ly += 20;
            });
        }, 1600);
    }

    /* ===================== SHAP BAR CHART ===================== */
    function drawSHAP() {
        const canvas = $('#shapCanvas');
        if (!canvas) return;
        const { ctx, w, h } = setupHiDPI(canvas);
        const c = chartColors();

        const features = [
            { name: 'Lactate', value: 0.185 },
            { name: 'MAP', value: 0.152 },
            { name: 'Heart Rate', value: 0.128 },
            { name: 'Respiratory Rate', value: 0.115 },
            { name: 'Temperature', value: 0.098 },
            { name: 'WBC', value: 0.087 },
            { name: 'Creatinine', value: 0.076 },
            { name: 'Bilirubin (total)', value: 0.063 },
            { name: 'Platelets', value: 0.054 },
            { name: 'SpO2', value: 0.048 },
        ];

        const pad = { t: 20, r: 30, l: 130, b: 40 };
        const plotW = w - pad.l - pad.r;
        const plotH = h - pad.t - pad.b;
        const maxVal = 0.2;
        const barH = plotH / features.length;
        const barGap = 8;

        // Grid
        for (let i = 0; i <= 4; i++) {
            const x = pad.l + (plotW / 4) * i;
            ctx.strokeStyle = c.grid; ctx.lineWidth = 0.5;
            ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
            ctx.fillStyle = c.text; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText((maxVal / 4 * i).toFixed(3), x, h - pad.b + 16);
        }
        ctx.fillStyle = c.text; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('Mean |SHAP Value|', pad.l + plotW / 2, h - 4);

        // Axis
        ctx.strokeStyle = c.axis; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.stroke();

        // Animate bars
        const duration = 1200;
        const start = performance.now();
        function tick(now) {
            const t = clamp((now - start) / duration, 0, 1);
            const eased = 1 - Math.pow(1 - t, 3);

            // Clear bars area
            ctx.clearRect(pad.l + 1, pad.t, plotW, plotH);
            // Re-draw grid lines on top of cleared area
            for (let i = 1; i <= 4; i++) {
                const x = pad.l + (plotW / 4) * i;
                ctx.strokeStyle = c.grid; ctx.lineWidth = 0.5;
                ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
            }

            features.forEach((f, idx) => {
                const y = pad.t + barH * idx + barGap / 2;
                const bh = barH - barGap;
                const bw = (f.value / maxVal) * plotW * eased;

                // Gradient bar: red → blue based on importance
                const ratio = 1 - idx / (features.length - 1);
                const grad = ctx.createLinearGradient(pad.l, 0, pad.l + bw, 0);
                if (ratio > 0.5) {
                    grad.addColorStop(0, `rgba(239,68,68,${0.6 + ratio * 0.4})`);
                    grad.addColorStop(1, `rgba(251,146,60,${0.6 + ratio * 0.3})`);
                } else {
                    grad.addColorStop(0, `rgba(102,126,234,${0.5 + ratio * 0.5})`);
                    grad.addColorStop(1, `rgba(56,189,248,${0.4 + ratio * 0.4})`);
                }

                // Bar
                ctx.fillStyle = grad;
                roundRect(ctx, pad.l, y, bw, bh, 4);
                ctx.fill();

                // Glow
                ctx.shadowColor = ratio > 0.5 ? 'rgba(239,68,68,0.3)' : 'rgba(102,126,234,0.3)';
                ctx.shadowBlur = 10;
                ctx.fillStyle = grad;
                roundRect(ctx, pad.l, y, bw, bh, 4);
                ctx.fill();
                ctx.shadowBlur = 0;

                // Value label
                if (eased > 0.5) {
                    ctx.fillStyle = 'rgba(255,255,255,0.8)';
                    ctx.font = '10px Inter, sans-serif';
                    ctx.textAlign = 'left';
                    ctx.fillText(f.value.toFixed(3), pad.l + bw + 8, y + bh / 2 + 4);
                }

                // Feature name
                ctx.fillStyle = 'rgba(255,255,255,0.7)';
                ctx.font = '11px Inter, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(f.name, pad.l - 10, y + bh / 2 + 4);
            });

            if (t < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    /* ===================== CROSS-HOSPITAL CHART ===================== */
    function drawCrossHospital() {
        const canvas = $('#crossHospitalCanvas');
        if (!canvas) return;
        const { ctx, w, h } = setupHiDPI(canvas);
        const c = chartColors();

        const data = [
            { label: 'A→A', auc: 0.91, f1: 0.82, sens: 0.87 },
            { label: 'A→B', auc: 0.85, f1: 0.74, sens: 0.80 },
            { label: 'B→A', auc: 0.84, f1: 0.73, sens: 0.79 },
            { label: 'B→B', auc: 0.89, f1: 0.79, sens: 0.84 },
            { label: 'A+B→A', auc: 0.92, f1: 0.83, sens: 0.88 },
            { label: 'A+B→B', auc: 0.90, f1: 0.81, sens: 0.86 },
        ];

        const pad = { t: 30, r: 30, l: 50, b: 50 };
        const plotW = w - pad.l - pad.r;
        const plotH = h - pad.t - pad.b;
        const groupW = plotW / data.length;
        const barW = groupW * 0.22;
        const metrics = ['auc', 'f1', 'sens'];
        const colors = [c.accent1, c.accent2, c.accent3];
        const labels = ['AUC-ROC', 'F1', 'Sensitivity'];

        // Y-axis
        for (let i = 0; i <= 5; i++) {
            const val = 0.6 + i * 0.08;
            const y = pad.t + plotH - ((val - 0.6) / 0.4) * plotH;
            ctx.strokeStyle = c.grid; ctx.lineWidth = 0.5;
            ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
            ctx.fillStyle = c.text; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
            ctx.fillText(val.toFixed(2), pad.l - 8, y + 4);
        }
        ctx.strokeStyle = c.axis; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();

        // X labels
        data.forEach((d, i) => {
            const cx = pad.l + groupW * i + groupW / 2;
            ctx.fillStyle = c.text; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText(d.label, cx, h - pad.b + 20);
        });

        // Animate bars
        const duration = 1200;
        const startT = performance.now();
        function tick(now) {
            const t = clamp((now - startT) / duration, 0, 1);
            const eased = 1 - Math.pow(1 - t, 3);

            ctx.clearRect(pad.l + 1, pad.t, plotW, plotH);
            // Re-draw gridlines
            for (let i = 0; i <= 5; i++) {
                const val = 0.6 + i * 0.08;
                const y = pad.t + plotH - ((val - 0.6) / 0.4) * plotH;
                ctx.strokeStyle = c.grid; ctx.lineWidth = 0.5;
                ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
            }

            data.forEach((d, i) => {
                const cx = pad.l + groupW * i + groupW / 2;
                metrics.forEach((m, mi) => {
                    const val = d[m];
                    const barH = ((val - 0.6) / 0.4) * plotH * eased;
                    const bx = cx + (mi - 1) * (barW + 3) - barW / 2;
                    const by = pad.t + plotH - barH;

                    const grad = ctx.createLinearGradient(0, by, 0, pad.t + plotH);
                    grad.addColorStop(0, colors[mi]);
                    grad.addColorStop(1, colors[mi].replace(')', ',0.3)').replace('rgb', 'rgba').replace('#', ''));
                    // Simpler approach for hex
                    ctx.globalAlpha = 0.85;
                    ctx.fillStyle = colors[mi];
                    roundRect(ctx, bx, by, barW, barH, 3);
                    ctx.fill();
                    ctx.globalAlpha = 1;

                    // Value on top
                    if (eased > 0.7) {
                        ctx.fillStyle = 'rgba(255,255,255,0.7)';
                        ctx.font = '9px Inter, sans-serif';
                        ctx.textAlign = 'center';
                        ctx.fillText(val.toFixed(2), bx + barW / 2, by - 5);
                    }
                });
            });

            if (t < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);

        // Legend
        setTimeout(() => {
            labels.forEach((lbl, i) => {
                const lx = pad.l + 10 + i * 100;
                ctx.fillStyle = colors[i];
                roundRect(ctx, lx, pad.t + 5, 12, 12, 2);
                ctx.fill();
                ctx.fillStyle = c.text;
                ctx.font = '10px Inter, sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(lbl, lx + 18, pad.t + 14);
            });
        }, 1400);
    }

    /* ===================== DONUT CHARTS ===================== */
    function drawDonut(canvasId, positive, negative, animDelay) {
        const canvas = $(canvasId);
        if (!canvas) return;
        const { ctx, w, h } = setupHiDPI(canvas);
        const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 20;
        const total = positive + negative;
        const posAngle = (positive / total) * Math.PI * 2;

        const duration = 1200;
        const start = performance.now();
        function tick(now) {
            const t = clamp((now - start) / duration, 0, 1);
            const eased = 1 - Math.pow(1 - t, 3);

            ctx.clearRect(0, 0, w, h);

            // Negative slice
            ctx.beginPath();
            ctx.arc(cx, cy, r, -Math.PI / 2 + posAngle * eased, -Math.PI / 2 + Math.PI * 2 * eased);
            ctx.arc(cx, cy, r * 0.6, -Math.PI / 2 + Math.PI * 2 * eased, -Math.PI / 2 + posAngle * eased, true);
            ctx.closePath();
            ctx.fillStyle = 'rgba(56,189,248,0.3)';
            ctx.fill();

            // Positive slice
            ctx.beginPath();
            ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + posAngle * eased);
            ctx.arc(cx, cy, r * 0.6, -Math.PI / 2 + posAngle * eased, -Math.PI / 2, true);
            ctx.closePath();
            ctx.fillStyle = 'rgba(239,68,68,0.5)';
            ctx.fill();

            // Ring border
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255,255,255,0.06)';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(cx, cy, r * 0.6, 0, Math.PI * 2);
            ctx.stroke();

            if (t < 1) requestAnimationFrame(tick);
        }
        setTimeout(() => requestAnimationFrame(tick), animDelay || 0);
    }

    function drawDonuts() {
        drawDonut('#donutBefore', 2932, 37068, 0);
        drawDonut('#donutAfter', 37068, 37068, 400);
    }

    /* ===================== IMBALANCE STRATEGY CHART ===================== */
    function drawImbalance() {
        const canvas = $('#imbalanceCanvas');
        if (!canvas) return;
        const { ctx, w, h } = setupHiDPI(canvas);
        const c = chartColors();

        const data = [
            { label: 'No SMOTE', auc: 0.88, f1: 0.71, recall: 0.74, color: c.red },
            { label: 'SMOTE', auc: 0.91, f1: 0.82, recall: 0.87, color: c.green },
            { label: 'Focal Loss', auc: 0.90, f1: 0.80, recall: 0.85, color: '#a78bfa' },
            { label: 'Weighted BCE', auc: 0.89, f1: 0.79, recall: 0.84, color: c.blue },
        ];

        const pad = { t: 30, r: 30, l: 50, b: 50 };
        const plotW = w - pad.l - pad.r;
        const plotH = h - pad.t - pad.b;
        const groupW = plotW / data.length;
        const barW = groupW * 0.22;
        const metrics = ['auc', 'f1', 'recall'];
        const mLabels = ['AUC', 'F1', 'Recall'];
        const mColors = [c.accent1, c.accent2, c.accent3];

        // Y-axis
        for (let i = 0; i <= 5; i++) {
            const val = 0.6 + i * 0.08;
            const y = pad.t + plotH - ((val - 0.6) / 0.4) * plotH;
            ctx.strokeStyle = c.grid; ctx.lineWidth = 0.5;
            ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
            ctx.fillStyle = c.text; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
            ctx.fillText(val.toFixed(2), pad.l - 8, y + 4);
        }
        ctx.strokeStyle = c.axis; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();

        data.forEach((d, i) => {
            const cx = pad.l + groupW * i + groupW / 2;
            ctx.fillStyle = c.text; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText(d.label, cx, h - pad.b + 20);
        });

        const duration = 1200;
        const startT = performance.now();
        function tick(now) {
            const t = clamp((now - startT) / duration, 0, 1);
            const eased = 1 - Math.pow(1 - t, 3);

            ctx.clearRect(pad.l + 1, pad.t, plotW, plotH);
            for (let i = 0; i <= 5; i++) {
                const val = 0.6 + i * 0.08;
                const y = pad.t + plotH - ((val - 0.6) / 0.4) * plotH;
                ctx.strokeStyle = c.grid; ctx.lineWidth = 0.5;
                ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
            }

            data.forEach((d, i) => {
                const cx = pad.l + groupW * i + groupW / 2;
                metrics.forEach((m, mi) => {
                    const val = d[m];
                    const barH = ((val - 0.6) / 0.4) * plotH * eased;
                    const bx = cx + (mi - 1) * (barW + 3) - barW / 2;
                    const by = pad.t + plotH - barH;

                    ctx.globalAlpha = 0.85;
                    ctx.fillStyle = mColors[mi];
                    roundRect(ctx, bx, by, barW, barH, 3);
                    ctx.fill();
                    ctx.globalAlpha = 1;

                    if (eased > 0.7) {
                        ctx.fillStyle = 'rgba(255,255,255,0.7)';
                        ctx.font = '9px Inter, sans-serif';
                        ctx.textAlign = 'center';
                        ctx.fillText(val.toFixed(2), bx + barW / 2, by - 5);
                    }
                });
            });

            if (t < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);

        setTimeout(() => {
            mLabels.forEach((lbl, i) => {
                const lx = pad.l + 10 + i * 80;
                ctx.fillStyle = mColors[i];
                roundRect(ctx, lx, pad.t + 5, 12, 12, 2);
                ctx.fill();
                ctx.fillStyle = c.text;
                ctx.font = '10px Inter, sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(lbl, lx + 18, pad.t + 14);
            });
        }, 1400);
    }

    /* ===================== SMOOTH SCROLL ===================== */
    $$('a[href^="#"]').forEach(a => {
        a.addEventListener('click', e => {
            e.preventDefault();
            const target = $(a.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    /* ===================== INIT ===================== */
    // Force trigger hero animations immediately
    window.addEventListener('load', () => {
        $$('#hero .animate-in').forEach(el => {
            const delay = el.dataset.delay || 0;
            setTimeout(() => el.classList.add('visible'), +delay + 200);
        });
        // Stat bars
        $$('#hero .stat-bar-fill').forEach(bar => {
            setTimeout(() => { bar.style.width = bar.dataset.width + '%'; }, 800);
        });
        // Stat counters
        $$('#hero .stat-value[data-target]').forEach((sv, i) => {
            setTimeout(() => animateCounter(sv), 600 + i * 100);
        });
    });

})();

/* =====================================================
   PATIENT DEMO — Live Risk Assessment Module
   ===================================================== */

// ---- Heuristic risk model (mimics BiLSTM learned weights) ----
function computeRiskScore(v) {
    let score = 0.08; // baseline

    // Lactate — strongest sepsis marker (SHAP weight ~0.22)
    const lactateFactor = v.Lactate / 2.0;
    score += clampD(0, (v.Lactate - 2) * 0.055, 0.22);

    // MAP — hemodynamic compromise
    if (v.MAP < 65)  score += 0.18;
    else if (v.MAP < 75) score += 0.08;
    else if (v.MAP > 105) score += 0.02;

    // Heart Rate — tachycardia
    if (v.HR > 120) score += 0.12;
    else if (v.HR > 100) score += 0.06;
    else if (v.HR < 50) score += 0.04;

    // Respiration — tachypnea
    if (v.Resp > 25) score += 0.10;
    else if (v.Resp > 20) score += 0.04;

    // O2Sat — hypoxia
    if (v.O2Sat < 90) score += 0.14;
    else if (v.O2Sat < 94) score += 0.07;

    // Temperature — fever or hypothermia
    if (v.Temp > 38.3) score += 0.04;
    if (v.Temp > 39.5) score += 0.06;
    if (v.Temp < 36.0) score += 0.08;

    // Creatinine — renal failure
    if (v.Creatinine > 4) score += 0.10;
    else if (v.Creatinine > 2) score += 0.05;

    // WBC — leukocytosis or leukopenia
    if (v.WBC > 20 || v.WBC < 4) score += 0.07;
    else if (v.WBC > 12) score += 0.03;

    // pH — acidosis
    if (v.pH < 7.20) score += 0.14;
    else if (v.pH < 7.30) score += 0.07;
    else if (v.pH > 7.50) score += 0.03;

    return Math.min(0.97, Math.max(0.02, score));
}

function clampD(lo, v, hi) { return Math.min(hi, Math.max(lo, v)); }

// ---- Compute SHAP-like feature contributions ----
function computeShapContributions(v) {
    const baseline = 0.08;
    const contribs = [
        { name: 'Lactate',     val: clampD(-0.22, (v.Lactate - 2) * 0.055, 0.22) },
        { name: 'MAP',         val: v.MAP < 65 ? 0.18 : v.MAP < 75 ? 0.08 : v.MAP > 105 ? 0.02 : 0 },
        { name: 'Heart Rate',  val: v.HR > 120 ? 0.12 : v.HR > 100 ? 0.06 : v.HR < 50 ? 0.04 : 0 },
        { name: 'Resp. Rate',  val: v.Resp > 25 ? 0.10 : v.Resp > 20 ? 0.04 : 0 },
        { name: 'O2 Sat',      val: v.O2Sat < 90 ? 0.14 : v.O2Sat < 94 ? 0.07 : clampD(-0.04, (v.O2Sat - 97) * -0.01, 0) },
        { name: 'pH',          val: v.pH < 7.20 ? 0.14 : v.pH < 7.30 ? 0.07 : v.pH > 7.50 ? 0.03 : clampD(-0.03, (v.pH - 7.4) * -0.05, 0) },
        { name: 'Temperature', val: v.Temp < 36.0 ? 0.08 : v.Temp > 39.5 ? 0.10 : v.Temp > 38.3 ? 0.04 : 0 },
        { name: 'Creatinine',  val: v.Creatinine > 4 ? 0.10 : v.Creatinine > 2 ? 0.05 : 0 },
        { name: 'WBC',         val: (v.WBC > 20 || v.WBC < 4) ? 0.07 : v.WBC > 12 ? 0.03 : 0 },
    ];
    // Sort by |contribution|
    contribs.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
    return contribs.slice(0, 7);
}

// ---- qSOFA, NEWS2, SOFA ----
function computeClinicalScores(v) {
    // qSOFA (0–3)
    let qsofa = 0;
    if (v.Resp >= 22) qsofa++;
    if (v.MAP < 100) qsofa++; // SBP proxy using MAP
    // mentation: not available, skip

    // NEWS2 (0–20)
    let news2 = 0;
    if (v.Resp <= 8) news2 += 3; else if (v.Resp <= 11) news2 += 1; else if (v.Resp <= 20) news2 += 0; else if (v.Resp <= 24) news2 += 2; else news2 += 3;
    if (v.O2Sat <= 91) news2 += 3; else if (v.O2Sat <= 93) news2 += 2; else if (v.O2Sat <= 95) news2 += 1;
    if (v.MAP <= 70) news2 += 3; else if (v.MAP <= 80) news2 += 2; else if (v.MAP <= 90) news2 += 1;
    if (v.HR <= 40) news2 += 3; else if (v.HR <= 50) news2 += 1; else if (v.HR <= 90) news2 += 0; else if (v.HR <= 110) news2 += 1; else if (v.HR <= 130) news2 += 2; else news2 += 3;
    if (v.Temp <= 35.0) news2 += 3; else if (v.Temp <= 36.0) news2 += 1; else if (v.Temp <= 38.0) news2 += 0; else if (v.Temp <= 39.0) news2 += 1; else news2 += 2;

    // Partial SOFA (0–20, only available features)
    let sofa = 0;
    if (v.MAP < 65) sofa += 1;
    if (v.Creatinine > 5) sofa += 3; else if (v.Creatinine > 3.4) sofa += 2; else if (v.Creatinine > 2) sofa += 1;
    if (v.WBC < 2 || v.WBC > 20) sofa += 1;
    // pH proxy for respiratory
    if (v.pH < 7.2) sofa += 2; else if (v.pH < 7.3) sofa += 1;

    return { qsofa, news2, sofa };
}

// ---- Gauge arc helpers ----
function polarToXY(cx, cy, r, angleDeg) {
    const rad = (angleDeg - 90) * Math.PI / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx, cy, r, startAngle, endAngle) {
    const s = polarToXY(cx, cy, r, startAngle);
    const e = polarToXY(cx, cy, r, endAngle);
    const la = (endAngle - startAngle) > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${la} 1 ${e.x} ${e.y}`;
}

function updateGauge(score) {
    const ARC_START = -150;  // degrees (left end of arc)
    const ARC_RANGE = 300;   // total arc span
    const cx = 100, cy = 110, r = 80;

    const arcEl = document.getElementById('gaugeArcActive');
    const needleEl = document.getElementById('gaugeNeedle');
    const scoreText = document.getElementById('gaugeScoreText');
    const levelText = document.getElementById('gaugeLevelText');
    if (!arcEl) return;

    const angleDeg = ARC_START + score * ARC_RANGE;
    arcEl.setAttribute('d', describeArc(cx, cy, r, ARC_START, angleDeg));

    // Needle: rotate from left (-90°) to right (+90°)
    const needleRot = ARC_START + score * ARC_RANGE;
    needleEl.setAttribute('transform', `rotate(${needleRot} ${cx} ${cy})`);

    scoreText.textContent = score.toFixed(2);
}

function updateRiskBadge(score) {
    const badge = document.getElementById('riskLevelBadge');
    const interp = document.getElementById('riskInterpretation');
    const card = document.getElementById('riskGaugeCard');
    const gaugeLevel = document.getElementById('gaugeLevelText');
    if (!badge) return;

    badge.className = 'risk-level-badge';
    let level, emoji, text, color;
    if (score < 0.30) {
        level = 'low'; emoji = '🟢'; text = 'LOW RISK';
        color = 'rgba(34,197,94,0.08)';
        interp.textContent = 'Vitals within acceptable range. Continue standard monitoring.';
    } else if (score < 0.55) {
        level = 'moderate'; emoji = '🟡'; text = 'MODERATE RISK';
        color = 'rgba(234,179,8,0.08)';
        interp.textContent = 'Borderline indicators detected. Increase monitoring frequency, re-assess labs.';
    } else if (score < 0.78) {
        level = 'high'; emoji = '🔴'; text = 'HIGH RISK';
        color = 'rgba(249,115,22,0.1)';
        interp.textContent = 'Multiple abnormal parameters. Initiate sepsis bundle assessment immediately.';
    } else {
        level = 'critical'; emoji = '💀'; text = 'CRITICAL';
        color = 'rgba(239,68,68,0.12)';
        interp.textContent = '⚠️ SEPSIS ALERT — Immediate intervention required. Activate rapid response.';
    }
    badge.classList.add(level);
    badge.textContent = emoji + ' ' + text;
    gaugeLevel.textContent = text;
    card.style.background = `linear-gradient(135deg, ${color}, rgba(0,210,255,0.04))`;
}

function updateClinicalScores(scores) {
    const qsofaEl = document.getElementById('qsofaScore');
    const news2El = document.getElementById('news2Score');
    const sofaEl  = document.getElementById('sofaScore');
    const qfill = document.getElementById('qsofaFill');
    const nfill = document.getElementById('news2Fill');
    const sfill = document.getElementById('sofaFill');
    if (!qsofaEl) return;
    qsofaEl.textContent = `${scores.qsofa} / 3`;
    news2El.textContent = `${scores.news2} / 20`;
    sofaEl.textContent  = `${scores.sofa} / 8`;
    qfill.style.width  = `${(scores.qsofa / 3) * 100}%`;
    nfill.style.width  = `${(scores.news2 / 20) * 100}%`;
    sfill.style.width  = `${(scores.sofa / 8) * 100}%`;
    // Color coding
    const toColor = (pct) => pct > 0.6 ? 'linear-gradient(90deg,#ef4444,#f97316)' : pct > 0.35 ? 'linear-gradient(90deg,#eab308,#f97316)' : 'linear-gradient(90deg,#667eea,#a78bfa)';
    qfill.style.background = toColor(scores.qsofa / 3);
    nfill.style.background = toColor(scores.news2 / 20);
    sfill.style.background = toColor(scores.sofa / 8);
}

function renderShapWaterfall(contribs) {
    const container = document.getElementById('shapWaterfall');
    if (!container) return;
    const maxAbsVal = Math.max(...contribs.map(c => Math.abs(c.val)), 0.01);
    container.innerHTML = contribs.map(c => {
        const pct = (Math.abs(c.val) / maxAbsVal) * 45; // max 45% from centre
        const isPos = c.val >= 0;
        const sign = isPos ? '+' : '';
        return `<div class="shap-bar-row">
            <div class="shap-feat-name">${c.name}</div>
            <div class="shap-bar-track">
                <div class="shap-center-line"></div>
                <div class="shap-bar-fill ${isPos ? 'positive' : 'negative'}" style="width:${pct}%"></div>
            </div>
            <div class="shap-val ${isPos ? 'pos' : 'neg'}">${sign}${c.val.toFixed(3)}</div>
        </div>`;
    }).join('');
}

function renderAttentionHeatmap(score) {
    const canvas = document.getElementById('attentionHeatmap');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const N = 48; // time steps
    const cellW = W / N;

    // Simulate attention weights: more attention near end of window when high risk
    for (let i = 0; i < N; i++) {
        const t = i / N;
        // Base pattern: recent hours get more attention
        let attn = Math.pow(t, 1.5) * 0.5;
        // With higher risk, attention spikes more at end
        attn += t > 0.75 ? score * 0.5 * (t - 0.75) / 0.25 : 0;
        // Small random variation
        attn += (Math.sin(i * 0.7 + score * 10) + 1) * 0.05;
        attn = Math.min(1, attn);

        // Map to color: dark purple → bright cyan/red based on score
        const r = Math.round(lerp(20, 239, attn * score));
        const g = Math.round(lerp(20, lerp(130, 68, score), attn));
        const b = Math.round(lerp(60, lerp(255, 68, score * 0.8), attn));
        ctx.fillStyle = `rgba(${r},${g},${b},${0.3 + attn * 0.7})`;
        ctx.fillRect(i * cellW, 0, cellW - 1, H);
    }
    // Gradient overlay
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(10,10,26,0)');
    grad.addColorStop(1, 'rgba(10,10,26,0.3)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);
}

function lerp(a, b, t) { return a + (b - a) * Math.max(0, Math.min(1, t)); }

// ---- Main update function ----
function updateDemo() {
    const vals = {
        HR:         parseFloat(document.getElementById('slider-HR')?.value || 85),
        O2Sat:      parseFloat(document.getElementById('slider-O2Sat')?.value || 97),
        Temp:       parseFloat(document.getElementById('slider-Temp')?.value || 37.0),
        MAP:        parseFloat(document.getElementById('slider-MAP')?.value || 85),
        Resp:       parseFloat(document.getElementById('slider-Resp')?.value || 16),
        Lactate:    parseFloat(document.getElementById('slider-Lactate')?.value || 1.2),
        Creatinine: parseFloat(document.getElementById('slider-Creatinine')?.value || 1.0),
        WBC:        parseFloat(document.getElementById('slider-WBC')?.value || 8.0),
        pH:         parseFloat(document.getElementById('slider-pH')?.value || 7.40),
    };

    // Update display values
    const fmt = (v, dec) => parseFloat(v).toFixed(dec);
    document.getElementById('val-HR').textContent     = fmt(vals.HR, 0);
    document.getElementById('val-O2Sat').textContent  = fmt(vals.O2Sat, 0);
    document.getElementById('val-Temp').textContent   = fmt(vals.Temp, 1);
    document.getElementById('val-MAP').textContent    = fmt(vals.MAP, 0);
    document.getElementById('val-Resp').textContent   = fmt(vals.Resp, 0);
    document.getElementById('val-Lactate').textContent    = fmt(vals.Lactate, 1);
    document.getElementById('val-Creatinine').textContent = fmt(vals.Creatinine, 1);
    document.getElementById('val-WBC').textContent    = fmt(vals.WBC, 1);
    document.getElementById('val-pH').textContent     = fmt(vals.pH, 2);

    // Highlight sliders out of normal range
    colorSlider('slider-HR', vals.HR, 60, 100);
    colorSlider('slider-O2Sat', vals.O2Sat, 95, 100);
    colorSlider('slider-Temp', vals.Temp, 36.1, 38.0);
    colorSlider('slider-MAP', vals.MAP, 70, 100);
    colorSlider('slider-Resp', vals.Resp, 12, 20);
    colorSlider('slider-Lactate', vals.Lactate, 0, 2.0);
    colorSlider('slider-Creatinine', vals.Creatinine, 0.7, 1.2);
    colorSlider('slider-WBC', vals.WBC, 4, 11);
    colorSlider('slider-pH', vals.pH, 7.35, 7.45);

    const score = computeRiskScore(vals);
    updateGauge(score);
    updateRiskBadge(score);
    updateClinicalScores(computeClinicalScores(vals));
    renderShapWaterfall(computeShapContributions(vals));
    renderAttentionHeatmap(score);
}

function colorSlider(id, val, lo, hi) {
    const el = document.getElementById(id);
    const label = document.getElementById('val-' + id.replace('slider-', ''));
    if (!el || !label) return;
    const ok = val >= lo && val <= hi;
    const pct = (val - el.min) / (el.max - el.min);
    const color = ok ? '#667eea' : (val < lo ? '#3b82f6' : '#ef4444');
    el.style.background = `linear-gradient(90deg, ${color} ${pct*100}%, rgba(255,255,255,0.08) ${pct*100}%)`;
    label.style.color = ok ? '#a78bfa' : (val < lo ? '#60a5fa' : '#f87171');
}

// ---- Preset patients ----
function applyPreset(type) {
    const presets = {
        healthy: { HR:72, O2Sat:98, Temp:36.8, MAP:88, Resp:14, Lactate:1.0, Creatinine:0.9, WBC:7.5, pH:7.42 },
        early:   { HR:102, O2Sat:94, Temp:38.5, MAP:72, Resp:22, Lactate:2.8, Creatinine:1.5, WBC:14.0, pH:7.35 },
        sepsis:  { HR:118, O2Sat:91, Temp:39.2, MAP:62, Resp:28, Lactate:4.2, Creatinine:2.8, WBC:18.5, pH:7.27 },
        critical:{ HR:136, O2Sat:86, Temp:35.5, MAP:48, Resp:36, Lactate:8.5, Creatinine:5.2, WBC:2.5,  pH:7.15 },
    };
    const p = presets[type];
    if (!p) return;

    // Animate slider values
    Object.entries(p).forEach(([key, target]) => {
        const slider = document.getElementById('slider-' + key);
        if (!slider) return;
        const start = parseFloat(slider.value);
        const dur = 600;
        const t0 = performance.now();
        const step = (now) => {
            const frac = Math.min(1, (now - t0) / dur);
            const ease = 1 - Math.pow(1 - frac, 3);
            slider.value = start + (target - start) * ease;
            if (frac < 1) requestAnimationFrame(step); else updateDemo();
        };
        requestAnimationFrame(step);
    });

    // Mark active preset button
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    const ids = { healthy:'presetHealthy', early:'presetEarly', sepsis:'presetSepsis', critical:'presetCritical' };
    const btn = document.getElementById(ids[type]);
    if (btn) btn.classList.add('active');

    // Debounced final update
    setTimeout(updateDemo, 700);
}

// Init demo on page load
window.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('slider-HR')) {
        updateDemo();
    }
});

