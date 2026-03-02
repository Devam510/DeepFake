/**
 * DeepFake Detector — Frontend JS
 * Supports both image and video analysis.
 * All dynamic: every value shown comes directly from actual model outputs.
 */

// ── Tab switching (global so onclick="switchTab()" can call it) ─────────────
let currentTab = 'image';

function switchTab(tab) {
    currentTab = tab;
    const isVideo = tab === 'video';

    document.getElementById('tab-image').classList.toggle('active', !isVideo);
    document.getElementById('tab-video').classList.toggle('active', isVideo);
    document.getElementById('upload-panel-image').classList.toggle('hidden', isVideo);
    document.getElementById('upload-panel-video').classList.toggle('hidden', !isVideo);
    document.getElementById('pill-image-models').classList.toggle('hidden', isVideo);
    document.getElementById('pill-video-models').classList.toggle('hidden', !isVideo);
}

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ──────────────────────────────────────────────────
    // Image
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewSection = document.getElementById('preview-section');
    const previewImage = document.getElementById('preview-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');

    // Video
    const uploadAreaVideo = document.getElementById('upload-area-video');
    const videoInput = document.getElementById('video-input');
    const videoBrowseLink = document.getElementById('video-browse-link');
    const videoPreviewSection = document.getElementById('video-preview-section');
    const previewVideo = document.getElementById('preview-video');
    const analyzeVideoBtn = document.getElementById('analyze-video-btn');
    const clearVideoBtn = document.getElementById('clear-video-btn');

    // Shared
    const uploadSection = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');

    let selectedFile = null;
    let selectedVideoFile = null;

    // ── IMAGE upload interactions ─────────────────────────────────
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => e.target.files[0] && handleImageFile(e.target.files[0]));

    uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) handleImageFile(f);
    });

    function handleImageFile(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            previewImage.src = e.target.result;
            uploadArea.classList.add('hidden');
            previewSection.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    clearBtn.addEventListener('click', resetImageUpload);
    function resetImageUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewSection.classList.add('hidden');
        uploadArea.classList.remove('hidden');
    }

    // ── VIDEO upload interactions ─────────────────────────────────
    uploadAreaVideo.addEventListener('click', () => videoInput.click());
    if (videoBrowseLink) videoBrowseLink.addEventListener('click', e => { e.stopPropagation(); videoInput.click(); });
    videoInput.addEventListener('change', e => e.target.files[0] && handleVideoFile(e.target.files[0]));

    uploadAreaVideo.addEventListener('dragover', e => { e.preventDefault(); uploadAreaVideo.classList.add('dragover'); });
    uploadAreaVideo.addEventListener('dragleave', () => uploadAreaVideo.classList.remove('dragover'));
    uploadAreaVideo.addEventListener('drop', e => {
        e.preventDefault();
        uploadAreaVideo.classList.remove('dragover');
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('video/')) handleVideoFile(f);
    });

    function handleVideoFile(file) {
        selectedVideoFile = file;
        const url = URL.createObjectURL(file);
        previewVideo.src = url;
        uploadAreaVideo.classList.add('hidden');
        videoPreviewSection.classList.remove('hidden');
    }

    clearVideoBtn.addEventListener('click', resetVideoUpload);
    function resetVideoUpload() {
        selectedVideoFile = null;
        videoInput.value = '';
        previewVideo.src = '';
        videoPreviewSection.classList.add('hidden');
        uploadAreaVideo.classList.remove('hidden');
    }

    // ── IMAGE Analyze ─────────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing...';
        analyzeBtn.querySelector('.btn-loader').classList.remove('hidden');

        try {
            const fd = new FormData();
            fd.append('image', selectedFile);

            const res = await fetch('/api/detect', { method: 'POST', body: fd });
            const data = await res.json();

            if (data.success) {
                renderResults(data, previewImage.src, 'image');
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error(err);
            alert('Failed to analyze image. Please try again.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('.btn-text').textContent = '🔎 Analyze Image';
            analyzeBtn.querySelector('.btn-loader').classList.add('hidden');
        }
    });

    // ── VIDEO Analyze ─────────────────────────────────────────────
    analyzeVideoBtn.addEventListener('click', async () => {
        if (!selectedVideoFile) return;

        analyzeVideoBtn.disabled = true;
        analyzeVideoBtn.querySelector('.btn-text').textContent = 'Analyzing... (this may take 10-30s)';
        analyzeVideoBtn.querySelector('.btn-loader').classList.remove('hidden');

        try {
            const fd = new FormData();
            fd.append('video', selectedVideoFile);

            const res = await fetch('/api/detect-video', { method: 'POST', body: fd });
            const data = await res.json();

            if (data.success) {
                renderResults(data, previewVideo.src, 'video');
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error(err);
            alert('Failed to analyze video. Please try again.');
        } finally {
            analyzeVideoBtn.disabled = false;
            analyzeVideoBtn.querySelector('.btn-text').textContent = '🔎 Analyze Video';
            analyzeVideoBtn.querySelector('.btn-loader').classList.add('hidden');
        }
    });

    // ── Render results ────────────────────────────────────────────
    function renderResults(data, mediaSrc, mediaType) {
        uploadSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Filename
        document.getElementById('result-filename').textContent =
            (data.filename || 'Unknown') + (data.elapsed_seconds ? ` · ${data.elapsed_seconds}s` : '');

        // Show correct media element
        const resultImg = document.getElementById('result-image');
        const resultVideo = document.getElementById('result-video');
        const videoInfoBadge = document.getElementById('video-info-badge');

        if (mediaType === 'video') {
            resultImg.classList.add('hidden');
            resultVideo.src = mediaSrc;
            resultVideo.classList.remove('hidden');

            // Video info badge
            const vi = data.video_info || {};
            if (vi.duration && videoInfoBadge) {
                videoInfoBadge.classList.remove('hidden');
                videoInfoBadge.innerHTML =
                    `🎬 ${vi.duration}s &nbsp;·&nbsp; ${vi.frames_analyzed} frames &nbsp;·&nbsp; ${vi.resolution || ''}`;
            }

            // Update raw score chips for video
            updateVideoChips(data.raw_scores || {});
        } else {
            resultVideo.classList.add('hidden');
            if (videoInfoBadge) videoInfoBadge.classList.add('hidden');
            resultImg.src = mediaSrc;
            resultImg.classList.remove('hidden');
            updateImageChips(data.raw_scores || {});
        }

        // AI probability score (animated)
        animateNum(document.getElementById('overall-score'), 0, data.overall_score, 1100);
        setTimeout(() => {
            const fill = document.getElementById('score-bar-fill');
            fill.style.width = data.overall_score + '%';
            fill.style.background = scoreColor(data.overall_score);
        }, 100);

        // CI text
        const ci = data.confidence_interval;
        document.getElementById('ci-text').textContent =
            ci ? `Confidence interval: ${ci.lower}% – ${ci.upper}%` : '';

        // Verdict pill
        const pill = document.getElementById('verdict-pill');
        const icon = document.getElementById('verdict-icon');
        const vtext = document.getElementById('verdict-text');
        const vcard = document.getElementById('verdict-card');

        pill.className = 'verdict-pill';
        vcard.className = 'verdict-card';

        const v = data.verdict || 'UNCERTAIN';
        if (v === 'AI-GENERATED') {
            pill.classList.add('v-ai'); icon.textContent = '🤖'; vtext.textContent = 'AI-GENERATED';
            vcard.classList.add('ai-glow');
        } else if (v === 'LIKELY AI-GENERATED') {
            pill.classList.add('v-ai'); icon.textContent = '⚠️'; vtext.textContent = 'LIKELY AI-GENERATED';
            vcard.classList.add('ai-glow');
        } else if (v === 'LIKELY REAL') {
            pill.classList.add('v-real'); icon.textContent = '📷'; vtext.textContent = 'LIKELY REAL';
            vcard.classList.add('real-glow');
        } else if (v === 'REAL') {
            pill.classList.add('v-real'); icon.textContent = '✅'; vtext.textContent = 'REAL';
            vcard.classList.add('real-glow');
        } else if (v === 'POSSIBLY AI') {
            pill.classList.add('v-maybe'); icon.textContent = '🔍'; vtext.textContent = 'POSSIBLY AI';
            vcard.classList.add('uncertain-glow');
        } else {
            pill.classList.add('v-unknown'); icon.textContent = '❓'; vtext.textContent = 'UNCERTAIN';
            vcard.classList.add('uncertain-glow');
        }

        document.getElementById('confidence-text').textContent = data.confidence || '--';

        // Meta tiles
        document.getElementById('meta-source').textContent = data.decision_source || '—';

        if (mediaType === 'video') {
            document.getElementById('meta-processing').textContent = `🎬 ${data.video_info?.frames_analyzed || '?'} frames`;
            document.getElementById('meta-filter').textContent = '🎞️ Video Analysis';
        } else {
            const proc = data.image_processing_level || 'unknown';
            const procMap = { minimal_processing: '✅ Minimal', moderate_processing: '⚠️ Moderate', heavy_processing: '🔴 Heavy', unknown: '— Unknown' };
            document.getElementById('meta-processing').textContent = procMap[proc] || proc;
            document.getElementById('meta-filter').textContent = data.filter_detected
                ? `${data.filter_type} (${data.filter_confidence}%)`
                : '✅ None detected';
        }

        const disag = data.model_disagreement ?? 0;
        const metaAgree = document.getElementById('meta-agreement');
        if (disag > 45) {
            metaAgree.textContent = `⚠️ ${disag}% disagreement`;
            metaAgree.style.color = 'var(--yellow)';
        } else {
            metaAgree.textContent = `${100 - disag}% agreement`;
            metaAgree.style.color = disag < 20 ? 'var(--green)' : 'inherit';
        }

        // Warning banner
        const wb = document.getElementById('warning-banner');
        if (data.processing_warning) {
            wb.classList.remove('hidden');
            document.getElementById('warning-text').textContent = data.processing_warning;
        } else {
            wb.classList.add('hidden');
        }

        // Summary
        document.getElementById('summary-text').textContent = data.summary || '';

        // Signals grid
        const grid = document.getElementById('signals-grid');
        grid.innerHTML = '';
        const signals = data.signals || [];
        signals.forEach((sig, i) => {
            const score = sig.score ?? null;
            const status = sig.status || 'unknown';
            const scoreLabel = score !== null ? score + '%' : 'N/A';

            const card = document.createElement('div');
            card.className = 'signal-card';
            card.style.animationDelay = (i * 0.04) + 's';
            card.innerHTML = `
                <div class="signal-top">
                    <div>
                        <div class="signal-name-row">
                            <span class="signal-icon">${sig.icon || '📌'}</span>
                            <span class="signal-name">${sig.name}</span>
                        </div>
                        <div class="signal-source">${sig.source || ''}</div>
                    </div>
                    <div class="signal-score-chip ${status}">${scoreLabel}</div>
                </div>
                <div class="signal-bar-track">
                    <div class="signal-bar-fill ${status}" style="width:0%" data-target="${score ?? 0}"></div>
                </div>
                <p class="signal-explanation">${sig.explanation || ''}</p>
            `;
            grid.appendChild(card);
        });

        setTimeout(() => {
            document.querySelectorAll('.signal-bar-fill').forEach(el => {
                el.style.width = (el.dataset.target || 0) + '%';
            });
        }, 150);

        // Limitations
        const limList = document.getElementById('limit-list');
        limList.innerHTML = '';
        (data.limitations || []).forEach(l => {
            const li = document.createElement('li');
            li.textContent = l;
            limList.appendChild(li);
        });
    }

    function updateImageChips(rs) {
        document.getElementById('chip-eff-label').textContent = 'EfficientNet';
        document.getElementById('chip-stat-label').textContent = 'Statistical';
        document.getElementById('chip-forensic-label').textContent = 'Forensic';
        document.getElementById('chip-voter-label').textContent = 'Meta-Voter';
        document.getElementById('raw-eff').textContent = rs.efficientnet != null ? rs.efficientnet + '%' : '—';
        document.getElementById('raw-stat').textContent = rs.statistical != null ? rs.statistical + '%' : '—';
        document.getElementById('raw-forensic').textContent = rs.forensic != null ? rs.forensic + '%' : '—';
        document.getElementById('raw-voter').textContent = rs.meta_voter === 'active' ? '✅ Active' : (rs.meta_voter === 'fallback' ? '⚠️ Fallback' : '—');
    }

    function updateVideoChips(rs) {
        document.getElementById('chip-eff-label').textContent = 'Frame AI';
        document.getElementById('chip-stat-label').textContent = 'Temporal';
        document.getElementById('chip-forensic-label').textContent = 'Biological';
        document.getElementById('chip-voter-label').textContent = 'Audio';
        document.getElementById('raw-eff').textContent = rs.frame_ai != null ? rs.frame_ai + '%' : '—';
        document.getElementById('raw-stat').textContent = rs.temporal != null ? rs.temporal + '%' : '—';
        document.getElementById('raw-forensic').textContent = rs.biological != null ? rs.biological + '%' : '—';
        document.getElementById('raw-voter').textContent = rs.audio != null ? rs.audio + '%' : '—';
    }

    // ── New Analysis Buttons ──────────────────────────────────────
    document.getElementById('new-analysis-btn').addEventListener('click', startOver);
    document.getElementById('analyze-another-btn').addEventListener('click', startOver);

    function startOver() {
        resetImageUpload();
        resetVideoUpload();
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        // Restore whichever tab was last used
        switchTab(currentTab);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // ── Utilities ─────────────────────────────────────────────────
    function scoreColor(s) {
        if (s > 60) return 'var(--red)';
        if (s > 40) return 'var(--yellow)';
        return 'var(--green)';
    }

    function animateNum(el, from, to, ms) {
        const start = performance.now();
        function tick(now) {
            const t = Math.min((now - start) / ms, 1);
            const v = Math.floor(from + (to - from) * easeOut(t));
            el.innerHTML = v + '<span class="ai-prob-unit">%</span>';
            if (t < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    function easeOut(x) { return 1 - Math.pow(1 - x, 4); }
});
