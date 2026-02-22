/**
 * DeepFake Detection - Frontend JavaScript
 * Handles file upload, API interaction, and results display
 * Displays exact same data as CLI ensemble_predict()
 */

document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewSection = document.getElementById('preview-section');
    const previewImage = document.getElementById('preview-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultsSection = document.getElementById('results-section');
    const uploadSection = document.getElementById('upload-section');

    let selectedFile = null;

    // Upload Area Click
    uploadArea.addEventListener('click', () => fileInput.click());

    // File Input Change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFileSelect(file);
    });

    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileSelect(file);
        }
    });

    // Handle File Selection
    function handleFileSelect(file) {
        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadArea.classList.add('hidden');
            previewSection.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // Clear Button
    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewSection.classList.add('hidden');
        uploadArea.classList.remove('hidden');
    });

    // Analyze Button
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing...';
        analyzeBtn.querySelector('.btn-loader').classList.remove('hidden');

        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('image', selectedFile);

            // Call API
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data, previewImage.src);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze image. Please try again.');
        } finally {
            // Reset button
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('.btn-text').textContent = 'Analyze Image';
            analyzeBtn.querySelector('.btn-loader').classList.add('hidden');
        }
    });

    // Display Results — maps ALL ensemble verdict types
    function displayResults(data, imageSrc) {
        // Hide upload section, show results
        uploadSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Set result image
        document.getElementById('result-image').src = imageSrc;

        // Image type
        document.getElementById('image-type').innerHTML =
            `Image: <span>${data.filename || 'Unknown'}</span>`;

        // Overall score (animate)
        const scoreValue = document.getElementById('overall-score');
        const scoreBarFill = document.getElementById('score-bar-fill');
        animateNumber(scoreValue, 0, data.overall_score, 1000);
        setTimeout(() => {
            scoreBarFill.style.width = data.overall_score + '%';

            // Color the bar based on score
            if (data.overall_score > 60) {
                scoreBarFill.style.background = 'var(--red)';
            } else if (data.overall_score > 40) {
                scoreBarFill.style.background = 'var(--yellow)';
            } else {
                scoreBarFill.style.background = 'var(--green)';
            }
        }, 100);

        // Confidence interval
        const ciText = document.getElementById('confidence-interval-text');
        if (data.confidence_interval) {
            ciText.textContent = `Confidence Interval: ${data.confidence_interval.lower}% – ${data.confidence_interval.upper}%`;
        }

        // Verdict badge — handle ALL ensemble verdicts
        const verdictBadge = document.getElementById('verdict-badge');
        const verdictText = document.getElementById('verdict-text');
        verdictBadge.className = 'verdict-badge';

        const verdict = data.verdict || 'UNCERTAIN';

        if (verdict === 'AI-GENERATED') {
            verdictBadge.classList.add('ai');
            verdictBadge.querySelector('.verdict-icon').textContent = '🤖';
            verdictText.textContent = 'AI-GENERATED';
        } else if (verdict === 'LIKELY REAL') {
            verdictBadge.classList.add('real');
            verdictBadge.querySelector('.verdict-icon').textContent = '📷';
            verdictText.textContent = 'LIKELY REAL';
        } else if (verdict === 'POSSIBLY AI') {
            verdictBadge.classList.add('possibly-ai');
            verdictBadge.querySelector('.verdict-icon').textContent = '🔍';
            verdictText.textContent = 'POSSIBLY AI';
        } else if (verdict === 'UNCERTAIN') {
            verdictBadge.classList.add('uncertain');
            verdictBadge.querySelector('.verdict-icon').textContent = '❓';
            verdictText.textContent = 'UNCERTAIN';
        } else {
            // Fallback for any other verdict string
            verdictBadge.classList.add('uncertain');
            verdictBadge.querySelector('.verdict-icon').textContent = '❓';
            verdictText.textContent = verdict;
        }

        // Confidence
        document.getElementById('confidence-text').textContent = data.confidence || '--';

        // Domain routing info
        const domainText = document.getElementById('domain-text');
        const domain = data.detected_domain || 'unknown';
        const domainLabels = {
            'face': '👤 Face',
            'non_face_photo': '🖼️ Non-Face Photo',
            'art_or_illustration': '🎨 Art/Illustration',
            'synthetic_graphics': '📐 Synthetic Graphics',
            'unknown': '— Unknown'
        };
        const domainLabel = domainLabels[domain] || domain;
        const domainConf = data.domain_confidence || 0;
        domainText.textContent = `${domainLabel} (${domainConf}%)`;

        // Decision source (detector used)
        document.getElementById('decision-source-text').textContent = data.decision_source || '--';

        // Processing level
        const processingText = document.getElementById('processing-level-text');
        const level = data.image_processing_level || 'unknown';
        const levelLabels = {
            'minimal_processing': '✅ Minimal',
            'moderate_processing': '⚠️ Moderate',
            'heavy_processing': '🔴 Heavy',
            'unknown': '— Unknown'
        };
        processingText.textContent = levelLabels[level] || level;

        // Processing warning
        const warningBanner = document.getElementById('processing-warning');
        if (data.processing_warning) {
            warningBanner.classList.remove('hidden');
            document.getElementById('processing-warning-text').textContent = data.processing_warning;
        } else {
            warningBanner.classList.add('hidden');
        }

        // Filter detection
        const filterCard = document.getElementById('filter-card');
        if (data.filter_detected) {
            filterCard.style.display = '';
            document.getElementById('filter-text').textContent =
                `${data.filter_type} (${data.filter_confidence}% confidence)`;
        } else {
            filterCard.style.display = 'none';
        }

        // Model disagreement
        const disagText = document.getElementById('disagreement-text');
        const disagVal = data.model_disagreement || 0;
        if (disagVal > 40) {
            disagText.textContent = `⚠️ ${disagVal}% disagreement`;
            disagText.classList.add('warning-text-color');
        } else {
            disagText.textContent = `${disagVal}% disagreement`;
            disagText.classList.remove('warning-text-color');
        }

        // Summary
        document.getElementById('summary-text').textContent = data.summary;

        // Breakdown
        const breakdownList = document.getElementById('breakdown-list');
        breakdownList.innerHTML = '';

        if (data.breakdown && data.breakdown.length > 0) {
            data.breakdown.forEach((item, index) => {
                const statusClass = item.status;

                const itemHTML = `
                    <div class="breakdown-item" style="animation-delay: ${index * 0.05}s">
                        <div class="breakdown-header">
                            <span class="breakdown-name">${item.name}</span>
                            <span class="breakdown-score ${statusClass}">${item.score}%</span>
                        </div>
                        <div class="breakdown-bar">
                            <div class="breakdown-bar-fill ${statusClass}" style="width: 0%"></div>
                        </div>
                        <p class="breakdown-explanation">${item.explanation}</p>
                    </div>
                `;
                breakdownList.innerHTML += itemHTML;
            });

            // Animate breakdown bars
            setTimeout(() => {
                const fills = document.querySelectorAll('.breakdown-bar-fill');
                data.breakdown.forEach((item, index) => {
                    if (fills[index]) {
                        fills[index].style.width = item.score + '%';
                    }
                });
            }, 200);
        }

        // Limitations (always shown)
        const limitationsList = document.getElementById('limitations-list');
        limitationsList.innerHTML = '';
        if (data.limitations && data.limitations.length > 0) {
            data.limitations.forEach(lim => {
                const li = document.createElement('li');
                li.textContent = lim;
                limitationsList.appendChild(li);
            });
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Number animation helper
    function animateNumber(element, start, end, duration) {
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const current = Math.floor(start + (end - start) * easeOutQuart(progress));

            element.innerHTML = current + '<span class="score-percent">%</span>';

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // Easing function
    function easeOutQuart(x) {
        return 1 - Math.pow(1 - x, 4);
    }

    // New Analysis Button
    document.getElementById('new-analysis-btn').addEventListener('click', () => {
        // Reset everything
        selectedFile = null;
        fileInput.value = '';
        previewSection.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
});
