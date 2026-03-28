document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadSection = document.getElementById('uploadSection');
    const uploadZone = document.getElementById('uploadZone');
    const imageInput = document.getElementById('imageInput');
    const browseBtn = document.getElementById('browseBtn');
    
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const uploadContent = document.getElementById('uploadContent');
    const removeBtn = document.getElementById('removeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const resetBtn = document.getElementById('resetBtn');
    
    // Results Elements
    const healthStatusBadge = document.getElementById('healthStatusBadge');
    const healthStatusText = document.getElementById('healthStatusText');
    const plantNameElement = document.getElementById('plantName');
    const diseaseNameElement = document.getElementById('diseaseName');
    const confidenceVal = document.getElementById('confidenceVal');
    const confidenceBar = document.getElementById('confidenceBar');
    const remediesContainer = document.getElementById('remediesContainer');

    // Toast
    const errorToast = document.getElementById('errorToast');
    const errorToastMessage = document.getElementById('errorToastMessage');

    let currentFile = null;

    // --- File Upload Handlers ---
    
    browseBtn.addEventListener('click', () => {
        imageInput.click();
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.remove('dragover');
        }, false);
    });

    uploadZone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        if (files.length) {
            handleFile(files[0]);
        }
    }, false);

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file.');
            return;
        }
        
        currentFile = file;
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        }
    }

    removeBtn.addEventListener('click', () => {
        currentFile = null;
        imageInput.value = '';
        imagePreview.src = '';
        uploadContent.classList.remove('hidden');
        previewContainer.classList.add('hidden');
    });

    resetBtn.addEventListener('click', () => {
        hideAllSections();
        removeBtn.click();
        uploadSection.classList.remove('hidden');
    });

    // --- API & Analysis ---
    
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        hideAllSections();
        loadingSection.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                displayResults(data.predictions);
            } else {
                showError(data.error || 'Prediction failed. Please try again.');
                resetToUpload();
            }
        } catch (error) {
            showError('Network error. Make sure the server is running.');
            resetToUpload();
        }
    });

    function displayResults(predictions) {
        if (!predictions || predictions.length === 0) {
            showError('No prediction returned.');
            resetToUpload();
            return;
        }

        const topPrediction = predictions[0];
        
        hideAllSections();
        resultsSection.classList.remove('hidden');

        // Set Main Texts
        plantNameElement.textContent = topPrediction.plant;
        diseaseNameElement.textContent = topPrediction.disease;
        
        // Confidence Bar
        const percentage = Math.round(topPrediction.percentage);
        confidenceVal.textContent = `${percentage}%`;
        // Delay setting the width so the CSS transition plays
        setTimeout(() => {
            confidenceBar.style.width = `${percentage}%`;
        }, 100);

        // Determine Status
        let isHealthyStr = topPrediction.disease.toLowerCase();
        let isHealthy = isHealthyStr.includes('healthy');
        if (topPrediction.remedy_info && topPrediction.remedy_info.is_healthy !== undefined) {
            isHealthy = topPrediction.remedy_info.is_healthy;
        }

        healthStatusBadge.className = 'status-badge'; // reset
        if (isHealthy) {
            healthStatusBadge.classList.add('status-healthy');
            healthStatusText.textContent = 'Healthy';
        } else {
            healthStatusBadge.classList.add('status-diseased');
            healthStatusText.textContent = 'Disease Detected';
        }

        // Render Remedies
        renderRemedies(topPrediction);
    }

    function renderRemedies(prediction) {
        remediesContainer.innerHTML = '';
        
        const info = prediction.remedy_info;
        if (!info) {
            if (prediction.disease.toLowerCase().includes('healthy')) {
                 remediesContainer.innerHTML = `
                    <div class="remedy-card">
                        <h4>Excellent Condition!</h4>
                        <p>No issues detected. Keep up your regular maintenance.</p>
                    </div>
                `;
            } else {
                remediesContainer.innerHTML = `
                    <div class="remedy-card">
                        <h4>No Remedy Data</h4>
                        <p>Detailed remedy information is not available for this specific condition.</p>
                    </div>
                `;
            }
            return;
        }

        let html = '';

        // Description
        if (info.description) {
            html += `
                <div class="remedy-card">
                    <h4>Description</h4>
                    <p>${info.description}</p>
                </div>
            `;
        }

        // Remedies loop
        if (info.remedies) {
            for (const [category, items] of Object.entries(info.remedies)) {
                if (items && items.length > 0) {
                    const title = category.replace('_', ' ');
                    let lis = items.map(i => `<li>${i}</li>`).join('');
                    html += `
                        <div class="remedy-card">
                            <h4>${title}</h4>
                            <ul class="remedy-list">
                                ${lis}
                            </ul>
                        </div>
                    `;
                }
            }
        }

        // Prevention
        if (info.prevention_tips && info.prevention_tips.length > 0) {
            let lis = info.prevention_tips.map(i => `<li>${i}</li>`).join('');
            html += `
                <div class="remedy-card">
                    <h4>Prevention Tips</h4>
                    <ul class="remedy-list">
                        ${lis}
                    </ul>
                </div>
            `;
        }

        remediesContainer.innerHTML = html;
    }

    // --- Utils ---

    function hideAllSections() {
        uploadSection.classList.add('hidden');
        loadingSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        confidenceBar.style.width = '0%'; // reset bar
    }

    function resetToUpload() {
        hideAllSections();
        uploadSection.classList.remove('hidden');
    }

    function showError(msg) {
        errorToastMessage.textContent = msg;
        errorToast.classList.remove('hidden');
        errorToast.classList.add('show');
        
        setTimeout(() => {
            errorToast.classList.remove('show');
            setTimeout(() => errorToast.classList.add('hidden'), 300);
        }, 4000);
    }
});
