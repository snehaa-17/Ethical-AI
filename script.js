// static/script.js

let currentMode = 'manual';
let autoInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    initSliders();

    // Default Mode
    setMode('manual');

    // Analyze Button (Manual)
    document.getElementById('analyzeBtn').addEventListener('click', () => {
        triggerAnalysis('manual');
    });

    // Reset Button
    document.getElementById('resetBtn').addEventListener('click', async () => {
        await fetch('/api/reset', { method: 'POST' });
        document.getElementById('dayDisplay').innerText = '0';
        document.getElementById('echoFeatures').innerHTML = '';
        document.getElementById('resultContent').classList.add('hidden');
    });
});

function initSliders() {
    const sliders = {
        'avg_daily_screen_time': { suffix: 'h' },
        'night_usage_ratio': { isPercent: true },
        'sleep_irregularity_score': { suffix: '' },
        'social_app_withdrawal_score': { suffix: '' },
        'typing_speed_variance': { suffix: 'ms' },
        'app_usage_diversity': { suffix: ' apps' }
    };

    Object.keys(sliders).forEach(id => {
        const slider = document.getElementById(id);
        const displayId = id.replace('avg_daily_', 'val_')
            .replace('night_usage_', 'val_night_')
            .replace('sleep_irregularity_score', 'val_sleep')
            .replace('social_app_withdrawal_score', 'val_withdrawal')
            .replace('typing_speed_', 'val_typing')
            .replace('app_usage_', 'val_app_div');

        const display = document.getElementById(displayId);

        slider.addEventListener('input', (e) => {
            let val = e.target.value;
            const config = sliders[id];
            if (config.isPercent) val = Math.round(val * 100) + '%';
            else val = val + config.suffix;
            if (display) display.innerText = val;
        });
    });
}

function setMode(mode) {
    currentMode = mode;

    // UI Toggle
    document.getElementById('btnModeManual').classList.toggle('active', mode === 'manual');
    document.getElementById('btnModeAuto').classList.toggle('active', mode === 'auto');

    const manualPanel = document.getElementById('manualControls');
    const autoPanel = document.getElementById('autoControls');
    const badge = document.getElementById('signalBadge');

    if (mode === 'manual') {
        manualPanel.classList.remove('hidden');
        autoPanel.classList.add('hidden');
        badge.innerText = "Manual Override";
        badge.style.background = "#e2e8f0";
        badge.style.color = "#475569";
        stopAutoLoop();
    } else {
        manualPanel.classList.add('hidden');
        autoPanel.classList.remove('hidden');
        badge.innerText = "Auto-Simulation Active";
        badge.style.background = "#dbeafe";
        badge.style.color = "#1e40af";
        startAutoLoop();
    }
}

function startAutoLoop() {
    if (autoInterval) clearInterval(autoInterval);
    // Poll every 3 seconds to simulate a "Day" passing
    triggerAnalysis('auto'); // Immediate trigger
    autoInterval = setInterval(() => {
        triggerAnalysis('auto');
    }, 3000);
}

function stopAutoLoop() {
    if (autoInterval) clearInterval(autoInterval);
}

async function triggerAnalysis(mode) {
    const payload = { mode: mode };

    if (mode === 'manual') {
        // Collect slider values
        payload.avg_daily_screen_time = document.getElementById('avg_daily_screen_time').value;
        payload.night_usage_ratio = document.getElementById('night_usage_ratio').value;
        payload.app_usage_diversity = document.getElementById('app_usage_diversity').value;
        payload.typing_speed_variance = document.getElementById('typing_speed_variance').value;
        payload.sleep_irregularity_score = document.getElementById('sleep_irregularity_score').value;
        payload.social_app_withdrawal_score = document.getElementById('social_app_withdrawal_score').value;
    }

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status === 'success') {
            updateUI(data);
        }
    } catch (err) {
        console.error(err);
    }
}

function updateUI(data) {
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('resultContent').classList.remove('hidden');

    // Update Auto-Mode Info
    if (data.mode === 'auto') {
        document.getElementById('dayDisplay').innerText = "Day " + data.day_index;

        // Visualize the incoming stream data briefly
        const f = data.input_echo;
        // f is [screen, night, div, type, sleep, social]
        // Just a quick summary string
        const validF = f || [];
        const summary = `Screen: ${validF[0]?.toFixed(1)}h | Night: ${(validF[1] * 100).toFixed(0)}% | SleepVar: ${validF[4]?.toFixed(2)}`;

        const list = document.getElementById('echoFeatures');
        const item = document.createElement('div');
        item.className = 'echo-item';
        item.innerText = summary;
        list.prepend(item);
        if (list.children.length > 5) list.lastChild.remove();
    }

    // --- Result Panel ---

    // Risk Label
    const rLabel = document.getElementById('riskLabel');
    rLabel.innerText = data.risk_level;
    rLabel.className = ''; // Reset
    if (data.risk_level === 'Low') rLabel.classList.add('text-low');
    else if (data.risk_level === 'Moderate') rLabel.classList.add('text-mod');
    else rLabel.classList.add('text-high');

    // Metrics
    document.getElementById('confVal').innerText = (data.confidence * 100).toFixed(0) + '%';
    document.getElementById('trendVal').innerText = data.trend;

    // Text
    document.getElementById('explanationText').innerHTML = data.explanation.replace(/\n/g, '<br>');
    document.getElementById('counterfactualText').innerHTML = data.counterfactual;

    // Bars
    const container = document.getElementById('barsContainer');
    container.innerHTML = '';

    const sorted = data.feature_data.sort((a, b) => b.importance - a.importance);
    const maxImp = sorted[0].importance;

    sorted.forEach(feat => {
        const width = (feat.importance / maxImp) * 100;
        const cleanName = feat.name.replace(/_/g, ' ').replace('avg daily ', '').replace('score', '');

        const div = document.createElement('div');
        div.className = 'bar-item';
        div.innerHTML = `
            <div class="bar-meta">
                <span>${cleanName}</span>
                <span class="imp-score">${(feat.importance * 100).toFixed(1)}</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${width}%"></div>
            </div>
        `;
        container.appendChild(div);
    });
}
